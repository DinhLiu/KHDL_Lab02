"""Parsing Pipeline Controller - Orchestrates hierarchical LaTeX parsing"""

import json
import random
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .file_gatherer import FileGatherer, GatheredPublication
from .latex_parser import LaTeXParser, DocumentElement
from .standardizer import Standardizer, StandardizationConfig
from .hierarchy_builder import HierarchyBuilder, DocumentHierarchy
from .deduplicator import ReferenceDeduplicator, VersionedHierarchyDeduplicator


@dataclass
class ParsingResult:
    """Result of parsing a single publication"""
    pub_id: str
    success: bool
    hierarchy: Optional[DocumentHierarchy] = None
    bibitems: List[Dict] = None
    error: Optional[str] = None
    statistics: Dict = None
    
    def __post_init__(self):
        self.bibitems = self.bibitems or []
        self.statistics = self.statistics or {}


class ParsingPipeline:
    """Main controller for hierarchical parsing: gather files → parse → standardize → build hierarchy
    
    Element IDs in hierarchy.json include the publication ID:
    - Format: {pub_id}_{type}_{counter}
    - Example: 2411-00230_section_0001
    
    This allows identification of which publication an element belongs to.
    """
    
    def __init__(self, base_path: str = None, config: StandardizationConfig = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config = config or StandardizationConfig()
        self.file_gatherer = FileGatherer()
        self.latex_parser = LaTeXParser()
        self.standardizer = Standardizer(self.config)
        self.hierarchy_builder = HierarchyBuilder()
    
    def process_publication(self, data_folder: str, pub_id: str, save_output: bool = True) -> ParsingResult:
        """Process a single publication through the pipeline"""
        pub_path = self.base_path / data_folder / pub_id
        
        if not pub_path.exists():
            return ParsingResult(pub_id=pub_id, success=False, error=f"Path not found: {pub_path}")
        
        try:
            publication = self.file_gatherer.gather_publication(pub_path)
            if not publication or not publication.tex_files:
                return ParsingResult(pub_id=pub_id, success=False, error="No TeX files found")
            
            versions = set(f.version for f in publication.tex_files)
            ref_dedup = ReferenceDeduplicator()
            # Pass pub_id to include in element IDs (format: {pub_id}_{type}_{counter})
            hierarchy_dedup = VersionedHierarchyDeduplicator(pub_id=pub_id)
            all_bibitems, main_hierarchy = [], None
            
            for version in sorted(versions, reverse=True):
                version_content = self._get_version_content(publication, version)
                if not version_content:
                    continue
                
                standardized = self.standardizer.standardize(version_content)
                hierarchy = self.hierarchy_builder.build(content=standardized, paper_id=pub_id)
                
                if main_hierarchy is None:
                    main_hierarchy = hierarchy
                
                hierarchy_dedup.add_hierarchy(hierarchy.to_output_format(), version)
                
                # Extract from \bibitem{} in .tex files
                version_bibitems = self.latex_parser.extract_bibitems(version_content)
                all_bibitems.extend(version_bibitems)
                ref_dedup.add_references(version_bibitems, version)
                
                # Also extract from .bib files for this version
                version_bib_files = [f for f in publication.bib_files if f.version == version]
                for bib_file in version_bib_files:
                    bib_entries = self.latex_parser.extract_bibtex_entries(bib_file.content)
                    all_bibitems.extend(bib_entries)
                    ref_dedup.add_references(bib_entries, version)
            
            if main_hierarchy is None:
                return ParsingResult(pub_id=pub_id, success=False, error="Could not build hierarchy")
            
            dedup_bibitems = ref_dedup.get_unique_references()
            
            # Filter out publications without any references (neither \bibitem nor .bib entries)
            if not dedup_bibitems:
                return ParsingResult(pub_id=pub_id, success=False, 
                                   error="No references found (neither \\bibitem nor .bib entries)")
            
            statistics = self._calculate_statistics(main_hierarchy, publication, dedup_bibitems)
            
            # Count references by source
            bibitem_count = sum(1 for b in all_bibitems if b.get('source') != 'bibtex')
            bibtex_count = sum(1 for b in all_bibitems if b.get('source') == 'bibtex')
            
            statistics.update({'versions_count': len(versions), 
                             'original_bibitems_count': len(all_bibitems),
                             'from_bibitem_count': bibitem_count,
                             'from_bibtex_count': bibtex_count,
                             'deduplicated_bibitems_count': len(dedup_bibitems)})
            
            if save_output:
                self._save_output(pub_id, main_hierarchy, dedup_bibitems, data_folder, 
                                 hierarchy_dedup, ref_dedup)
            
            return ParsingResult(pub_id=pub_id, success=True, hierarchy=main_hierarchy,
                               bibitems=dedup_bibitems, statistics=statistics)
        except Exception as e:
            return ParsingResult(pub_id=pub_id, success=False, error=str(e))
    
    def process_all(self, data_folder: str, limit: int = None, sample_size: int = None,
                    random_seed: int = 42, manual_pubs: set = None, save_output: bool = True,
                    num_workers: int = None) -> List[ParsingResult]:
        """Process all or sampled publications with parallel execution"""
        data_path = self.base_path / data_folder
        if not data_path.exists():
            return []
        
        all_pub_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        # Apply sampling if specified
        if sample_size is not None:
            manual_pubs = manual_pubs or set()
            manual_dirs = [d for d in all_pub_dirs if d.name in manual_pubs]
            non_manual_dirs = [d for d in all_pub_dirs if d.name not in manual_pubs]
            
            random.seed(random_seed)
            sampled = random.sample(non_manual_dirs, min(sample_size, len(non_manual_dirs)))
            pub_dirs = sorted(manual_dirs + sampled, key=lambda d: d.name)
            print(f"Sampling: {len(manual_dirs)} manual + {len(sampled)} random = {len(pub_dirs)} total")
        else:
            pub_dirs = all_pub_dirs
        
        if limit:
            pub_dirs = pub_dirs[:limit]
        
        # Parallel processing with ProcessPoolExecutor (faster than ThreadPoolExecutor for CPU-bound tasks)
        num_workers = num_workers or min(multiprocessing.cpu_count(), 8)
        print(f"Processing {len(pub_dirs)} publications with {num_workers} processes...")
        
        results = [None] * len(pub_dirs)
        pub_ids = [d.name for d in pub_dirs]
        
        # Prepare arguments for worker function (must be picklable)
        base_path_str = str(self.base_path)
        work_args = [(i, pub_id, data_folder, base_path_str, save_output) 
                     for i, pub_id in enumerate(pub_ids)]
        
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_one_publication, args) for args in work_args]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing"):
                idx, result = future.result()
                results[idx] = result
        
        # Print summary
        success_count = sum(1 for r in results if r.success)
        print(f"Completed: {success_count}/{len(results)} successful")
        
        return results
    
    def _get_version_content(self, publication: GatheredPublication, version: str) -> str:
        r"""Get merged content for a specific version, expanding \input commands."""
        return self.file_gatherer.merge_tex_content(publication, version)
    
    def _calculate_statistics(self, hierarchy: DocumentHierarchy, publication: GatheredPublication,
                             bibitems: List[Dict]) -> Dict:
        return {'tex_files_count': len(publication.tex_files), 'bib_files_count': len(publication.bib_files),
                'bibitems_count': len(bibitems), **hierarchy.to_dict().get('statistics', {})}
    
    def _bibitems_to_bibtex_string(self, bibitems: List[Dict]) -> str:
        """
        Convert deduplicated bibitems to BibTeX format string.
        
        Each entry uses the canonical_key and includes all merged fields.
        """
        entries = []
        
        for item in bibitems:
            bib = item.get('bibtex', {})
            if not bib:
                continue
            
            entry_type = bib.get('type', 'article')
            # Use canonical_key for deduplicated entries, or 'key' for regular
            key = item.get('canonical_key', bib.get('key', 'unknown'))
            
            fields = []
            for field in ['author', 'title', 'journal', 'booktitle', 'year', 
                          'volume', 'number', 'pages', 'doi', 'arxiv', 'url']:
                value = bib.get(field, '')
                if value:
                    # Escape special characters
                    value = str(value).replace('{', '\\{').replace('}', '\\}')
                    if field == 'arxiv':
                        fields.append(f'  eprint = {{{value}}}')
                        fields.append(f'  archiveprefix = {{arXiv}}')
                    else:
                        fields.append(f'  {field} = {{{value}}}')
            
            # Add note about alternative keys if deduplicated
            if 'all_keys' in item and len(item['all_keys']) > 1:
                alt_keys = [k for k in item['all_keys'] if k != key]
                if alt_keys:
                    fields.append(f'  note = {{Also cited as: {", ".join(alt_keys)}}}')
            
            if fields:
                entry = f'@{entry_type}{{{key},\n'
                entry += ',\n'.join(fields)
                entry += '\n}'
                entries.append(entry)
        
        return '\n\n'.join(entries)
    
    def _save_output(self, pub_id: str, hierarchy: DocumentHierarchy, bibitems: List[Dict], 
                     data_folder: str, hierarchy_dedup: VersionedHierarchyDeduplicator = None,
                     ref_dedup: 'ReferenceDeduplicator' = None):
        """Save parsing output files to output folder (preserves original data).
        
        Files saved to output/{data_folder}/{pub_id}/:
        - hierarchy.json: Deduplicated hierarchy with elements and version-specific structures
        - refs.bib: Unified BibTeX entries from deduplicated references
        - metadata.json: Copied from source
        - references.json: Copied from source
        """
        # Output to output/{data_folder}/{pub_id}/ instead of modifying source
        output_path = self.base_path / "output" / data_folder / pub_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Source path for copying metadata and references
        source_path = self.base_path / data_folder / pub_id
        
        # Copy metadata.json from source
        if (source_path / "metadata.json").exists():
            shutil.copy2(source_path / "metadata.json", output_path / "metadata.json")
        
        # Copy references.json from source
        if (source_path / "references.json").exists():
            shutil.copy2(source_path / "references.json", output_path / "references.json")
        
        # Save hierarchy.json with deduplicated content from all versions
        if hierarchy_dedup:
            merged_hierarchy = hierarchy_dedup.get_merged_output()
            
            # Apply citation key renaming if reference deduplication was done
            if ref_dedup:
                key_mapping = ref_dedup.get_key_mapping()
                merged_hierarchy = self._rename_citations_in_hierarchy(
                    merged_hierarchy, key_mapping
                )
            
            with open(output_path / "hierarchy.json", 'w', encoding='utf-8') as f:
                json.dump(merged_hierarchy, f, indent=2, ensure_ascii=False)
        else:
            hierarchy.save_output_format(output_path / "hierarchy.json")
        
        # Save refs.bib with unified BibTeX entries
        if bibitems:
            bibtex_content = self._bibitems_to_bibtex_string(bibitems)
            if bibtex_content:
                with open(output_path / "refs.bib", 'w', encoding='utf-8') as f:
                    f.write(bibtex_content)
    
    def _rename_citations_in_hierarchy(self, hierarchy: Dict, key_mapping: Dict[str, str]) -> Dict:
        """
        Rename citation keys in hierarchy elements to use canonical keys.
        
        When references are deduplicated, different citation keys may refer
        to the same reference. This updates \\cite{old_key} to \\cite{canonical_key}.
        """
        if not key_mapping:
            return hierarchy
        
        # Build regex pattern for all keys that need renaming
        keys_to_rename = {k: v for k, v in key_mapping.items() if k != v}
        if not keys_to_rename:
            return hierarchy
        
        import re
        
        def rename_in_text(text: str) -> str:
            """Rename citation keys in text"""
            for old_key, new_key in keys_to_rename.items():
                # Match \cite{...old_key...} patterns
                # Handle multiple citations: \cite{key1, key2}
                pattern = rf'(\\cite[a-z]*\{{[^}}]*)\b{re.escape(old_key)}\b([^}}]*\}})'
                text = re.sub(pattern, rf'\1{new_key}\2', text)
            return text
        
        # Update elements
        new_elements = {}
        for elem_id, content in hierarchy.get('elements', {}).items():
            new_elements[elem_id] = rename_in_text(content)
        
        return {
            **hierarchy,
            'elements': new_elements
        }
    
    def get_summary(self, results: List[ParsingResult]) -> Dict:
        successful = [r for r in results if r.success]
        return {
            'total_publications': len(results), 'successful': len(successful), 'failed': len(results) - len(successful),
            'total_bibitems': sum(len(r.bibitems) for r in successful),
            'total_nodes': sum(r.statistics.get('total_nodes', 0) for r in successful),
            'failed_publications': [{'pub_id': r.pub_id, 'error': r.error} for r in results if not r.success]
        }


def parse_publication(data_folder: str, pub_id: str, base_path: str = None) -> ParsingResult:
    """Convenience function to parse a single publication"""
    return ParsingPipeline(base_path=base_path).process_publication(data_folder, pub_id)


def _process_one_publication(args) -> Tuple[int, ParsingResult]:
    """Worker function for parallel processing (must be at module level for pickling)"""
    idx, pub_id, data_folder, base_path, save_output = args
    pipeline = ParsingPipeline(base_path=base_path)
    result = pipeline.process_publication(data_folder, pub_id, save_output)
    return idx, result


def parse_all_publications(data_folder: str, base_path: str = None, limit: int = None) -> Tuple[List[ParsingResult], Dict]:
    """Convenience function to parse all publications"""
    pipeline = ParsingPipeline(base_path=base_path)
    results = pipeline.process_all(data_folder, limit=limit)
    return results, pipeline.get_summary(results)


if __name__ == "__main__":
    
    
    BASE_PATH = Path(__file__).parent.parent.parent
    DATA_FOLDER = "23120260"
    OUTPUT_FOLDER = BASE_PATH / "output" / DATA_FOLDER
    
    # Manual labeled publications (must be included in sample)
    MANUAL_PUBS = {
        "2411-00222",  # test
        "2411-00223",  # valid  
        "2411-00225",  # train
        "2411-00226",  # train
        "2411-00227",  # train
    }
    
    SAMPLE_SIZE = 995
    
    print("=" * 50)
    print(f"Parsing Pipeline - Sampling {SAMPLE_SIZE} publications")
    print("=" * 50)
    print(f"Source: {BASE_PATH / DATA_FOLDER}")
    print(f"Output: {OUTPUT_FOLDER}")
    print(f"Manual pubs (always included): {len(MANUAL_PUBS)}")
    
    # Ensure output folder exists
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    pipeline = ParsingPipeline(base_path=BASE_PATH)
    results = pipeline.process_all(
        DATA_FOLDER, 
        sample_size=SAMPLE_SIZE,
        manual_pubs=MANUAL_PUBS,
        random_seed=42
    )
    summary = pipeline.get_summary(results)
    
    print(f"\nSummary: {summary['successful']}/{summary['total_publications']} successful")
    print(f"Bibitems: {summary['total_bibitems']}, Nodes: {summary['total_nodes']}")
    
    # Report failed publications (no deletion - source data preserved)
    failed_pubs = [r.pub_id for r in results if not r.success]
    if failed_pubs:
        print(f"\nFailed publications: {len(failed_pubs)}")
        # Group by error type
        from collections import Counter
        error_counts = Counter(r.error for r in results if not r.success)
        for error, count in error_counts.most_common():
            print(f"  - {error}: {count}")

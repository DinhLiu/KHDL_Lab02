"""Parsing Pipeline Controller - Orchestrates hierarchical LaTeX parsing"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

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
    """Main controller for hierarchical parsing: gather files → parse → standardize → build hierarchy"""
    
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
            hierarchy_dedup = VersionedHierarchyDeduplicator()
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
                version_bibitems = self.latex_parser.extract_bibitems(version_content)
                all_bibitems.extend(version_bibitems)
                ref_dedup.add_references(version_bibitems, version)
            
            if main_hierarchy is None:
                return ParsingResult(pub_id=pub_id, success=False, error="Could not build hierarchy")
            
            dedup_bibitems = ref_dedup.get_unique_references()
            statistics = self._calculate_statistics(main_hierarchy, publication, dedup_bibitems)
            statistics.update({'versions_count': len(versions), 
                             'original_bibitems_count': len(all_bibitems),
                             'deduplicated_bibitems_count': len(dedup_bibitems)})
            
            if save_output:
                self._save_output(pub_id, main_hierarchy, dedup_bibitems, data_folder)
            
            return ParsingResult(pub_id=pub_id, success=True, hierarchy=main_hierarchy,
                               bibitems=dedup_bibitems, statistics=statistics)
        except Exception as e:
            return ParsingResult(pub_id=pub_id, success=False, error=str(e))
    
    def process_all(self, data_folder: str, limit: int = None, sample_size: int = None,
                    random_seed: int = 42, manual_pubs: set = None, save_output: bool = True) -> List[ParsingResult]:
        """Process all or sampled publications"""
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
        
        results = []
        for idx, pub_dir in enumerate(pub_dirs, 1):
            print(f"[{idx}/{len(pub_dirs)}] {pub_dir.name}...", end=" ")
            result = self.process_publication(data_folder, pub_dir.name, save_output)
            print(f"({result.statistics.get('total_nodes', 0)} nodes)" if result.success else result.error)
            results.append(result)
        
        return results
    
    def _get_version_content(self, publication: GatheredPublication, version: str) -> str:
        """Get merged content for a specific version"""
        version_files = [f for f in publication.tex_files if f.version == version]
        if not version_files:
            return ""
        
        main_content = next((f.content for f in version_files if f.file_type == 'main'), None)
        if not main_content:
            sorted_files = sorted(version_files, key=lambda f: len(f.content), reverse=True)
            main_content = sorted_files[0].content if sorted_files else ""
        
        other_contents = [f.content for f in version_files if f.content != main_content]
        return main_content + "".join(f"\n\n{c}" for c in other_contents)
    
    def _calculate_statistics(self, hierarchy: DocumentHierarchy, publication: GatheredPublication,
                             bibitems: List[Dict]) -> Dict:
        return {'tex_files_count': len(publication.tex_files), 'bib_files_count': len(publication.bib_files),
                'bibitems_count': len(bibitems), **hierarchy.to_dict().get('statistics', {})}
    
    def _save_output(self, pub_id: str, hierarchy: DocumentHierarchy, bibitems: List[Dict], data_folder: str):
        output_path = self.base_path / data_folder / pub_id
        hierarchy.save_output_format(output_path / f"{pub_id}.json")
    
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


def parse_all_publications(data_folder: str, base_path: str = None, limit: int = None) -> Tuple[List[ParsingResult], Dict]:
    """Convenience function to parse all publications"""
    pipeline = ParsingPipeline(base_path=base_path)
    results = pipeline.process_all(data_folder, limit=limit)
    return results, pipeline.get_summary(results)


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parent.parent.parent
    DATA_FOLDER = "23120260"
    MANUAL_PUBS = {"2411-00222", "2411-00223", "2411-00225", "2411-00226", "2411-00227"}
    SAMPLE_SIZE = 1500
    
    print("=" * 50)
    print("Parsing Pipeline")
    print("=" * 50)
    
    pipeline = ParsingPipeline(base_path=BASE_PATH)
    results = pipeline.process_all(DATA_FOLDER, sample_size=SAMPLE_SIZE, random_seed=42, manual_pubs=MANUAL_PUBS)
    summary = pipeline.get_summary(results)
    
    print(f"\nSummary: {summary['successful']}/{summary['total_publications']} successful")
    print(f"Bibitems: {summary['total_bibitems']}, Nodes: {summary['total_nodes']}")

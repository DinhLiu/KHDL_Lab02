"""
Parsing Pipeline Controller

Main controller for the hierarchical parsing pipeline.
Orchestrates all parsing components:
- FileGatherer: Collects LaTeX source files
- LaTeXParser: Parses LaTeX into structured elements
- Standardizer: Normalizes and cleans content
- HierarchyBuilder: Builds document hierarchy
- ReferenceDeduplicator: Deduplicates references across versions

Example usage:
    from src.parsing.parsing import ParsingPipeline
    
    pipeline = ParsingPipeline(base_path="C:/Code/KHDL_Lab02_v2")
    result = pipeline.process_publication("23120260", "2411-00222")
    
    # Or process all publications
    results = pipeline.process_all("23120260")
"""

import json
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
        if self.bibitems is None:
            self.bibitems = []
        if self.statistics is None:
            self.statistics = {}


class ParsingPipeline:
    """
    Main controller for the hierarchical parsing pipeline.
    
    Orchestrates the complete parsing workflow:
    1. Gather files from publication directory
    2. Parse LaTeX content
    3. Standardize/normalize text
    4. Build hierarchical structure
    5. Extract bibliography items
    
    Example:
        pipeline = ParsingPipeline()
        result = pipeline.process_publication(
            data_folder="23120260",
            pub_id="2411-00222"
        )
        
        if result.success:
            print(f"Parsed {result.statistics['total_nodes']} nodes")
            result.hierarchy.save(output_path)
    """
    
    def __init__(self, base_path: str = None, config: StandardizationConfig = None):
        """
        Initialize the parsing pipeline.
        
        Args:
            base_path: Base path to the workspace (default: current directory)
            config: Standardization configuration (default: use defaults)
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.config = config or StandardizationConfig()
        
        # Initialize components
        self.file_gatherer = FileGatherer()
        self.latex_parser = LaTeXParser()
        self.standardizer = Standardizer(self.config)
        self.hierarchy_builder = HierarchyBuilder()
    
    def process_publication(
        self,
        data_folder: str,
        pub_id: str,
        save_output: bool = True
    ) -> ParsingResult:
        """
        Process a single publication through the parsing pipeline.
        
        Args:
            data_folder: Name of the data folder (e.g., "23120260")
            pub_id: Publication ID (e.g., "2411-00222")
            save_output: Whether to save output files (saves to data folder)
            
        Returns:
            ParsingResult object with hierarchy and statistics
        """
        pub_path = self.base_path / data_folder / pub_id
        
        if not pub_path.exists():
            return ParsingResult(
                pub_id=pub_id,
                success=False,
                error=f"Publication path not found: {pub_path}"
            )
        
        try:
            # Step 1: Gather files
            publication = self.file_gatherer.gather_publication(pub_path)
            
            if not publication or not publication.tex_files:
                return ParsingResult(
                    pub_id=pub_id,
                    success=False,
                    error="No TeX files found in publication"
                )
            
            # Step 2: Process each version separately and deduplicate
            versions = set(f.version for f in publication.tex_files)
            ref_dedup = ReferenceDeduplicator()
            hierarchy_dedup = VersionedHierarchyDeduplicator()
            
            all_bibitems = []
            main_hierarchy = None
            
            for version in sorted(versions, reverse=True):
                # Get content for this version
                version_content = self._get_version_content(publication, version)
                if not version_content:
                    continue
                
                # Standardize content
                standardized_content = self.standardizer.standardize(version_content)
                
                # Build hierarchy
                hierarchy = self.hierarchy_builder.build(
                    content=standardized_content,
                    paper_id=pub_id
                )
                
                # Use the latest version as main hierarchy
                if main_hierarchy is None:
                    main_hierarchy = hierarchy
                
                # Add to hierarchy deduplicator
                hierarchy_dedup.add_hierarchy(hierarchy.to_output_format(), version)
                
                # Extract bibitems for this version
                version_bibitems = self.latex_parser.extract_bibitems(version_content)
                all_bibitems.extend(version_bibitems)
                
                # Add to reference deduplicator
                ref_dedup.add_references(version_bibitems, version)
            
            if main_hierarchy is None:
                return ParsingResult(
                    pub_id=pub_id,
                    success=False,
                    error="Could not build hierarchy from any version"
                )
            
            # Get deduplicated references
            dedup_bibitems = ref_dedup.get_unique_references()
            
            # Calculate statistics
            statistics = self._calculate_statistics(main_hierarchy, publication, dedup_bibitems)
            statistics['versions_count'] = len(versions)
            statistics['original_bibitems_count'] = len(all_bibitems)
            statistics['deduplicated_bibitems_count'] = len(dedup_bibitems)
            
            # Save output if requested (directly to data folder)
            if save_output:
                self._save_output(pub_id, main_hierarchy, dedup_bibitems, data_folder)
            
            return ParsingResult(
                pub_id=pub_id,
                success=True,
                hierarchy=main_hierarchy,
                bibitems=dedup_bibitems,
                statistics=statistics
            )
            
        except Exception as e:
            return ParsingResult(
                pub_id=pub_id,
                success=False,
                error=str(e)
            )
    
    def process_all(
        self,
        data_folder: str,
        limit: int = None,
        save_output: bool = True
    ) -> List[ParsingResult]:
        """
        Process all publications in a data folder.
        
        Args:
            data_folder: Name of the data folder (e.g., "23120260")
            limit: Maximum number of publications to process (None = all)
            save_output: Whether to save output files (saves to data folder)
            
        Returns:
            List of ParsingResult objects
        """
        data_path = self.base_path / data_folder
        
        if not data_path.exists():
            return []
        
        results = []
        pub_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        if limit:
            pub_dirs = pub_dirs[:limit]
        
        total = len(pub_dirs)
        for idx, pub_dir in enumerate(pub_dirs, 1):
            pub_id = pub_dir.name
            print(f"[{idx}/{total}] Processing {pub_id}...", end=" ")
            
            result = self.process_publication(
                data_folder=data_folder,
                pub_id=pub_id,
                save_output=save_output
            )
            
            if result.success:
                print(f"({result.statistics.get('total_nodes', 0)} nodes, {len(result.bibitems)} bibitems)")
            else:
                print(f"{result.error}")
            
            results.append(result)
        
        return results
    
    def _merge_tex_content(self, publication: GatheredPublication) -> str:
        """Merge all TeX files into a single content string."""
        # Find main file first
        main_content = ""
        other_contents = []
        
        for tex_file in publication.tex_files:
            if tex_file.file_type == 'main':
                main_content = tex_file.content
            else:
                other_contents.append(tex_file.content)
        
        # If no main file found, use the largest file
        if not main_content:
            sorted_files = sorted(
                publication.tex_files,
                key=lambda f: len(f.content),
                reverse=True
            )
            if sorted_files:
                main_content = sorted_files[0].content
                other_contents = [f.content for f in sorted_files[1:]]
        
        # Combine contents
        combined = main_content
        for content in other_contents:
            combined += "\n\n" + content
        
        return combined
    
    def _get_version_content(self, publication: GatheredPublication, version: str) -> str:
        """Get merged content for a specific version."""
        version_files = [f for f in publication.tex_files if f.version == version]
        
        if not version_files:
            return ""
        
        # Find main file
        main_content = ""
        other_contents = []
        
        for tex_file in version_files:
            if tex_file.file_type == 'main':
                main_content = tex_file.content
            else:
                other_contents.append(tex_file.content)
        
        # If no main file, use largest
        if not main_content:
            sorted_files = sorted(version_files, key=lambda f: len(f.content), reverse=True)
            if sorted_files:
                main_content = sorted_files[0].content
                other_contents = [f.content for f in sorted_files[1:]]
        
        combined = main_content
        for content in other_contents:
            combined += "\n\n" + content
        
        return combined
    
    def _calculate_statistics(
        self,
        hierarchy: DocumentHierarchy,
        publication: GatheredPublication,
        bibitems: List[Dict]
    ) -> Dict:
        """Calculate statistics for the parsing result."""
        hierarchy_dict = hierarchy.to_dict()
        
        return {
            'tex_files_count': len(publication.tex_files),
            'bib_files_count': len(publication.bib_files),
            'bibitems_count': len(bibitems),
            **hierarchy_dict.get('statistics', {})
        }
    
    def _save_output(
        self,
        pub_id: str,
        hierarchy: DocumentHierarchy,
        bibitems: List[Dict],
        data_folder: str
    ):
        """Save parsing output directly to the data folder."""
        # Save directly to data folder: 23120260/2411-00222/<pub_id>.json
        output_path = self.base_path / data_folder / pub_id
        
        # Save in required format as <pub_id>.json
        hierarchy.save_output_format(output_path / f"{pub_id}.json")
    
    def get_summary(self, results: List[ParsingResult]) -> Dict:
        """
        Get summary statistics from a list of parsing results.
        
        Args:
            results: List of ParsingResult objects
            
        Returns:
            Summary dictionary with statistics
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_bibitems = sum(len(r.bibitems) for r in successful)
        total_nodes = sum(
            r.statistics.get('total_nodes', 0)
            for r in successful
        )
        
        return {
            'total_publications': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_bibitems': total_bibitems,
            'total_nodes': total_nodes,
            'failed_publications': [
                {'pub_id': r.pub_id, 'error': r.error}
                for r in failed
            ]
        }


# Convenience functions for simple usage
def parse_publication(
    data_folder: str,
    pub_id: str,
    base_path: str = None
) -> ParsingResult:
    """
    Convenience function to parse a single publication.
    
    Args:
        data_folder: Name of the data folder (e.g., "23120260")
        pub_id: Publication ID (e.g., "2411-00222")
        base_path: Base path to workspace
        
    Returns:
        ParsingResult object
    """
    pipeline = ParsingPipeline(base_path=base_path)
    return pipeline.process_publication(data_folder, pub_id)


def parse_all_publications(
    data_folder: str,
    base_path: str = None,
    limit: int = None
) -> Tuple[List[ParsingResult], Dict]:
    """
    Convenience function to parse all publications.
    
    Args:
        data_folder: Name of the data folder (e.g., "23120260")
        base_path: Base path to workspace
        limit: Maximum number of publications to process
        
    Returns:
        Tuple of (results list, summary dict)
    """
    pipeline = ParsingPipeline(base_path=base_path)
    results = pipeline.process_all(data_folder, limit=limit)
    summary = pipeline.get_summary(results)
    return results, summary


if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    # Default configuration
    BASE_PATH = Path(__file__).parent.parent.parent  # Go up to project root
    DATA_FOLDER = "23120260"
    
    print("=" * 60)
    print("Parsing Pipeline Controller")
    print("=" * 60)
    
    pipeline = ParsingPipeline(base_path=BASE_PATH)
    
    # Process a few publications as example
    print(f"\nProcessing publications from {DATA_FOLDER}...")
    results = pipeline.process_all(DATA_FOLDER
                                   #, limit=1000
                                   )
    
    summary = pipeline.get_summary(results)
    
    print(f"\nSummary:")
    print(f"  Total: {summary['total_publications']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Total bibitems extracted: {summary['total_bibitems']}")
    print(f"  Total hierarchy nodes: {summary['total_nodes']}")
    
    if summary['failed_publications']:
        print(f"\nFailed publications:")
        for fail in summary['failed_publications']:
            print(f"  - {fail['pub_id']}: {fail['error']}")

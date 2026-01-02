"""
Multi-file Gatherer for LaTeX Publications

Gathers all LaTeX source files from publication directories,
handling multiple versions and nested file structures.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TexFile:
    """Represents a gathered TeX file"""
    path: Path
    content: str
    version: str
    file_type: str  # 'main', 'chapter', 'appendix', 'bib', 'other'


@dataclass
class GatheredPublication:
    """Contains all gathered files from a publication"""
    pub_id: str
    pub_path: Path
    tex_files: List[TexFile]
    bib_files: List[TexFile]
    metadata: Dict


class FileGatherer:
    r"""
    Gathers all LaTeX source files from a publication directory.
    
    Handles:
    - Multiple version directories (v1, v2, etc.)
    - Nested file structures
    - \input{} and \include{} commands
    - .tex and .bib files
    
    Example usage:
        gatherer = FileGatherer()
        publication = gatherer.gather_publication(Path("23120260/2411-00222"))
        if publication:
            print(f"Found {len(publication.tex_files)} tex files")
    """
    
    # Patterns to identify main tex file
    MAIN_FILE_PATTERNS = [
        r'main\.tex',
        r'paper\.tex', 
        r'manuscript\.tex',
        r'article\.tex',
        r'\d{4}\.\d{5}.*\.tex'  # arXiv ID pattern
    ]
    
    def __init__(self):
        self.gathered_files = []
    
    def gather_publication(self, pub_path: Path) -> Optional[GatheredPublication]:
        """
        Gather all files from a publication directory.
        Only gathers files that are actually included in the compilation path.
        
        Args:
            pub_path: Path to publication directory
            
        Returns:
            GatheredPublication object or None if no tex files found
        """
        pub_id = pub_path.name
        tex_path = pub_path / 'tex'
        
        if not tex_path.exists():
            return None
        
        tex_files = []
        bib_files = []
        
        # Process each version directory
        for version_dir in sorted(tex_path.iterdir()):
            if not version_dir.is_dir():
                continue
            
            version = version_dir.name
            
            # First, find the main file and trace the include path
            all_tex_files = {}
            main_file_path = None
            
            for tex_file in version_dir.rglob('*.tex'):
                try:
                    content = tex_file.read_text(encoding='utf-8', errors='ignore')
                    file_type = self._classify_tex_file(tex_file, content)
                    
                    # Store with relative path from version_dir
                    rel_path = tex_file.relative_to(version_dir)
                    all_tex_files[str(rel_path)] = {
                        'path': tex_file,
                        'content': content,
                        'file_type': file_type
                    }
                    
                    # Track main file
                    if file_type == 'main' and main_file_path is None:
                        main_file_path = tex_file
                        
                except Exception as e:
                    continue
            
            # If no main file found, use the largest tex file
            if main_file_path is None and all_tex_files:
                largest = max(all_tex_files.items(), 
                             key=lambda x: len(x[1]['content']))
                main_file_path = largest[1]['path']
            
            # Trace included files starting from main file
            included_files = set()
            if main_file_path:
                main_content = all_tex_files.get(
                    str(main_file_path.relative_to(version_dir)), {}
                ).get('content', '')
                included_files = self._trace_includes(
                    main_content, version_dir, all_tex_files
                )
                # Always include main file
                included_files.add(str(main_file_path.relative_to(version_dir)))
            
            # Only add files that are in the compilation path
            for rel_path, file_info in all_tex_files.items():
                # Include if it's in the traced path OR if no includes were found
                # (some papers have all content in one file)
                if rel_path in included_files or not included_files:
                    tex_files.append(TexFile(
                        path=file_info['path'],
                        content=file_info['content'],
                        version=version,
                        file_type=file_info['file_type']
                    ))
            
            # Gather .bib files (always include all of them)
            for bib_file in version_dir.rglob('*.bib'):
                try:
                    content = bib_file.read_text(encoding='utf-8', errors='ignore')
                    bib_files.append(TexFile(
                        path=bib_file,
                        content=content,
                        version=version,
                        file_type='bib'
                    ))
                except Exception as e:
                    continue
        
        if not tex_files:
            return None
        
        # Load metadata if exists
        metadata = self._load_metadata(pub_path)
        
        return GatheredPublication(
            pub_id=pub_id,
            pub_path=pub_path,
            tex_files=tex_files,
            bib_files=bib_files,
            metadata=metadata
        )
    
    def _trace_includes(self, content: str, base_dir: Path, 
                        all_files: Dict[str, Dict]) -> set:
        """
        Trace all files included via \\input or \\include commands.
        
        Args:
            content: Content of the file to trace from
            base_dir: Base directory for resolving relative paths
            all_files: Dict of all available files
            
        Returns:
            Set of relative paths that are included
        """
        included = set()
        
        # Find all \input{} and \include{} commands
        patterns = [
            r'\\input\{([^}]+)\}',
            r'\\include\{([^}]+)\}',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                filename = match.group(1).strip()
                
                # Try different path variations
                variations = [
                    filename,
                    filename + '.tex',
                    filename.replace('.tex', ''),
                ]
                
                for var in variations:
                    # Check if file exists in our collection
                    if var in all_files:
                        included.add(var)
                        # Recursively trace includes in this file
                        sub_includes = self._trace_includes(
                            all_files[var]['content'], 
                            base_dir, 
                            all_files
                        )
                        included.update(sub_includes)
                        break
        
        return included
    
    def _classify_tex_file(self, file_path: Path, content: str) -> str:
        """Classify the type of tex file"""
        filename = file_path.name.lower()
        
        # Check for main file patterns
        for pattern in self.MAIN_FILE_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                return 'main'
        
        # Check content for documentclass (indicates main file)
        if r'\documentclass' in content:
            return 'main'
        
        # Check for appendix
        if 'appendix' in filename or r'\appendix' in content:
            return 'appendix'
        
        # Check for chapter/section file
        if any(x in filename for x in ['chapter', 'section', 'intro', 'conclusion']):
            return 'chapter'
        
        return 'other'
    
    def _load_metadata(self, pub_path: Path) -> Dict:
        """Load metadata.json if exists"""
        import json
        metadata_file = pub_path / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def find_main_file(self, publication: GatheredPublication) -> Optional[TexFile]:
        """Find the main tex file from gathered files"""
        main_files = [f for f in publication.tex_files if f.file_type == 'main']
        
        if main_files:
            # Prefer latest version
            return sorted(main_files, key=lambda x: x.version, reverse=True)[0]
        
        return publication.tex_files[0] if publication.tex_files else None
    
    def merge_tex_content(self, publication: GatheredPublication) -> str:
        r"""
        Merge all tex files into single content, resolving \input commands.
        
        Args:
            publication: GatheredPublication object
            
        Returns:
            Merged LaTeX content string
        """
        main_file = self.find_main_file(publication)
        if not main_file:
            return ""
        
        # Build file map for the same version
        file_map = {}
        for tex_file in publication.tex_files:
            if tex_file.version == main_file.version:
                # Map by filename (with and without .tex)
                name = tex_file.path.name
                file_map[name] = tex_file.content
                if name.endswith('.tex'):
                    file_map[name[:-4]] = tex_file.content
        
        # Recursively resolve includes
        return self._resolve_includes(main_file.content, file_map)
    
    def _resolve_includes(self, content: str, file_map: Dict[str, str], 
                          depth: int = 0) -> str:
        """Recursively resolve \\input and \\include commands"""
        if depth > 10:  # Prevent infinite recursion
            return content
        
        def replace_include(match):
            filename = match.group(1)
            # Add .tex if not present
            if not filename.endswith('.tex'):
                lookup_name = filename
            else:
                lookup_name = filename[:-4]
            
            # Try to find the file
            included_content = file_map.get(filename) or file_map.get(lookup_name) or ""
            
            if included_content:
                # Recursively resolve in included content
                return self._resolve_includes(included_content, file_map, depth + 1)
            
            return ""
        
        # Replace \input{...} and \include{...}
        content = re.sub(r'\\input\{([^}]+)\}', replace_include, content)
        content = re.sub(r'\\include\{([^}]+)\}', replace_include, content)
        
        return content


def gather_all_publications(data_dir: Path) -> List[GatheredPublication]:
    """
    Gather all publications from a data directory.
    
    Args:
        data_dir: Path to directory containing publication folders
        
    Returns:
        List of GatheredPublication objects
    """
    gatherer = FileGatherer()
    publications = []
    
    for pub_dir in sorted(data_dir.iterdir()):
        if not pub_dir.is_dir():
            continue
        
        # Skip if no tex folder
        if not (pub_dir / 'tex').exists():
            continue
        
        pub = gatherer.gather_publication(pub_dir)
        if pub:
            publications.append(pub)
    
    return publications

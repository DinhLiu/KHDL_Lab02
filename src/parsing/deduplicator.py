"""
Cross-Version Deduplication Module

Handles deduplication of:
1. Reference entries across different versions of the same publication
2. Full-text content elements across versions

Deduplication strategies:
- References: Unionize by matching content (not just keys)
- Content: Hash-based matching for exact duplicates
"""

import hashlib
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DeduplicatedReference:
    """A deduplicated reference entry"""
    canonical_key: str
    keys: Set[str]  # All keys that refer to this reference
    versions: Set[str]  # Versions where this reference appears
    raw_content: str
    bibtex: Dict
    content_hash: str


class ReferenceDeduplicator:
    """
    Deduplicates reference entries across versions.
    
    Two references are considered duplicates if:
    1. They have the same citation key, OR
    2. Their content similarity is above a threshold
    
    Example usage:
        dedup = ReferenceDeduplicator()
        
        # Add references from different versions
        dedup.add_references(v1_bibitems, version="v1")
        dedup.add_references(v2_bibitems, version="v2")
        
        # Get deduplicated list
        unique_refs = dedup.get_unique_references()
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.references: Dict[str, DeduplicatedReference] = {}  # hash -> reference
        self.key_to_hash: Dict[str, str] = {}  # key -> hash mapping
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash from normalized content"""
        # Normalize content for comparison
        normalized = content.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = normalized.strip()
        
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]
    
    def _extract_key_features(self, content: str) -> Dict[str, str]:
        """Extract key features for matching"""
        features = {}
        
        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            features['year'] = year_match.group(0)
        
        # Extract first author (usually first word or before comma)
        words = content.split()
        if words:
            features['first_author'] = words[0].strip('.,;:')
        
        # Extract DOI if present
        doi_match = re.search(r'10\.\d{4,}/[^\s,]+', content)
        if doi_match:
            features['doi'] = doi_match.group(0)
        
        # Extract arXiv ID
        arxiv_match = re.search(r'(\d{4}\.\d{4,5})', content)
        if arxiv_match:
            features['arxiv'] = arxiv_match.group(1)
        
        return features
    
    def _find_matching_reference(self, content: str, key: str) -> Optional[str]:
        """Find if a matching reference already exists"""
        content_hash = self._get_content_hash(content)
        
        # Exact hash match
        if content_hash in self.references:
            return content_hash
        
        # Check if key already mapped
        if key in self.key_to_hash:
            return self.key_to_hash[key]
        
        # Try feature-based matching
        features = self._extract_key_features(content)
        
        for ref_hash, ref in self.references.items():
            ref_features = self._extract_key_features(ref.raw_content)
            
            # Match by DOI (strongest signal)
            if features.get('doi') and features.get('doi') == ref_features.get('doi'):
                return ref_hash
            
            # Match by arXiv ID
            if features.get('arxiv') and features.get('arxiv') == ref_features.get('arxiv'):
                return ref_hash
            
            # Match by year + first author
            if (features.get('year') == ref_features.get('year') and
                features.get('first_author') and ref_features.get('first_author') and
                features['first_author'].lower() == ref_features['first_author'].lower()):
                # Additional content similarity check
                if self._content_similarity(content, ref.raw_content) > self.similarity_threshold:
                    return ref_hash
        
        return None
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple similarity between two contents"""
        # Normalize
        norm1 = set(re.sub(r'[^\w\s]', '', content1.lower()).split())
        norm2 = set(re.sub(r'[^\w\s]', '', content2.lower()).split())
        
        if not norm1 or not norm2:
            return 0.0
        
        intersection = len(norm1 & norm2)
        union = len(norm1 | norm2)
        
        return intersection / union if union > 0 else 0.0
    
    def add_references(self, bibitems: List[Dict], version: str):
        """
        Add references from a specific version.
        
        Args:
            bibitems: List of bibitem dictionaries from LaTeXParser
            version: Version identifier (e.g., "v1", "v2")
        """
        for item in bibitems:
            key = item.get('key', '')
            content = item.get('raw_content', '')
            bibtex = item.get('bibtex', {})
            
            if not content:
                continue
            
            # Check for existing match
            match_hash = self._find_matching_reference(content, key)
            
            if match_hash:
                # Update existing reference
                self.references[match_hash].keys.add(key)
                self.references[match_hash].versions.add(version)
                self.key_to_hash[key] = match_hash
                
                # Merge bibtex fields (prefer non-empty values)
                existing_bibtex = self.references[match_hash].bibtex
                for field, value in bibtex.items():
                    if value and not existing_bibtex.get(field):
                        existing_bibtex[field] = value
            else:
                # Create new reference
                content_hash = self._get_content_hash(content)
                
                self.references[content_hash] = DeduplicatedReference(
                    canonical_key=key,
                    keys={key},
                    versions={version},
                    raw_content=content,
                    bibtex=bibtex.copy() if bibtex else {},
                    content_hash=content_hash
                )
                self.key_to_hash[key] = content_hash
    
    def get_unique_references(self) -> List[Dict]:
        """
        Get list of unique, deduplicated references.
        
        Returns:
            List of reference dictionaries with merged information
        """
        result = []
        
        for ref in self.references.values():
            result.append({
                'canonical_key': ref.canonical_key,
                'all_keys': list(ref.keys),
                'versions': list(ref.versions),
                'raw_content': ref.raw_content,
                'bibtex': ref.bibtex,
                'content_hash': ref.content_hash
            })
        
        return result
    
    def get_key_mapping(self) -> Dict[str, str]:
        """
        Get mapping from all keys to canonical keys.
        
        Returns:
            Dict mapping each citation key to its canonical form
        """
        mapping = {}
        
        for ref in self.references.values():
            for key in ref.keys:
                mapping[key] = ref.canonical_key
        
        return mapping


class ContentDeduplicator:
    """
    Deduplicates content elements using content hashing.
    
    Used for full-text deduplication across versions.
    
    Example usage:
        dedup = ContentDeduplicator()
        
        # Get or create unique ID for content
        elem_id, is_new = dedup.get_or_create_id(content, "sentence")
        
        # Check if content is duplicate
        if dedup.is_duplicate(content):
            print("Already seen this content")
    """
    
    def __init__(self):
        self.content_hashes: Dict[str, str] = {}  # hash -> element_id
        self.counter = 0
    
    def get_hash(self, content: str) -> str:
        """Get MD5 hash of content"""
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', content).strip()
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]
    
    def get_or_create_id(self, content: str, prefix: str = "elem") -> Tuple[str, bool]:
        """
        Get existing ID or create new one for content.
        
        Args:
            content: Content to hash
            prefix: Prefix for ID
            
        Returns:
            Tuple of (element_id, is_new)
        """
        content_hash = self.get_hash(content)
        
        if content_hash in self.content_hashes:
            return self.content_hashes[content_hash], False
        
        self.counter += 1
        element_id = f"{prefix}_{self.counter:04d}"
        self.content_hashes[content_hash] = element_id
        
        return element_id, True
    
    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate"""
        content_hash = self.get_hash(content)
        return content_hash in self.content_hashes
    
    def reset(self):
        """Reset the deduplicator state"""
        self.content_hashes = {}
        self.counter = 0


class VersionedHierarchyDeduplicator:
    """
    Handles deduplication across multiple versions of a document hierarchy.
    
    Merges hierarchies from different versions, identifying:
    - Elements that are identical across versions
    - Elements that are unique to specific versions
    
    Example usage:
        dedup = VersionedHierarchyDeduplicator()
        dedup.add_hierarchy(v1_hierarchy, "v1")
        dedup.add_hierarchy(v2_hierarchy, "v2")
        
        merged = dedup.get_merged_output()
    """
    
    def __init__(self):
        self.elements: Dict[str, Dict] = {}  # element_id -> {content, versions, ...}
        self.hierarchies: Dict[str, Dict] = {}  # version -> hierarchy mapping
        self.content_dedup = ContentDeduplicator()
    
    def add_hierarchy(self, hierarchy_data: Dict, version: str):
        """
        Add a hierarchy from a specific version.
        
        Args:
            hierarchy_data: Output format hierarchy dict with 'elements' and 'hierarchy'
            version: Version identifier
        """
        elements = hierarchy_data.get('elements', {})
        hierarchy = hierarchy_data.get('hierarchy', {}).get('1', {})
        
        version_mapping = {}  # old_id -> new_id for this version
        
        for elem_id, content in elements.items():
            # Get or create deduplicated ID
            new_id, is_new = self.content_dedup.get_or_create_id(content, elem_id.split('_')[0])
            version_mapping[elem_id] = new_id
            
            if is_new:
                self.elements[new_id] = {
                    'content': content,
                    'versions': {version},
                    'original_ids': {version: elem_id}
                }
            else:
                self.elements[new_id]['versions'].add(version)
                self.elements[new_id]['original_ids'][version] = elem_id
        
        # Remap hierarchy
        remapped_hierarchy = {}
        for child_id, parent_id in hierarchy.items():
            new_child = version_mapping.get(child_id, child_id)
            new_parent = version_mapping.get(parent_id, parent_id)
            remapped_hierarchy[new_child] = new_parent
        
        self.hierarchies[version] = remapped_hierarchy
    
    def get_merged_output(self) -> Dict:
        """
        Get merged output with deduplicated elements.
        
        Returns:
            Dict with 'elements' and 'hierarchy' for each version
        """
        # Build elements dict
        elements = {
            elem_id: data['content']
            for elem_id, data in self.elements.items()
        }
        
        # Build hierarchy dict with versions as keys
        hierarchy = {}
        for version, hier in self.hierarchies.items():
            hierarchy[version] = hier
        
        return {
            'elements': elements,
            'hierarchy': hierarchy,
            'element_versions': {
                elem_id: list(data['versions'])
                for elem_id, data in self.elements.items()
            }
        }

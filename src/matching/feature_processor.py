"""
Feature extraction processor for multiprocessing.
This module must be importable by worker processes.
"""
import re
from typing import Dict, List, Set
import Levenshtein


class FeatureProcessor:
    """Feature extraction functions for parallel processing"""
    
    FEATURE_NAMES = [
        'title_jaccard', 'title_overlap', 'title_edit_dist',
        'author_overlap', 'first_author_match',
        'year_match', 'year_diff',
        'arxiv_match', 'arxiv_in_content',
        'num_matching_authors', 'title_len_ratio', 'combined_score'
    ]
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean title text"""
        if not title:
            return ""
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text"""
        if not text:
            return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    @staticmethod
    def extract_author_last_names(authors: List[str]) -> List[str]:
        """Extract last names from author list"""
        last_names = []
        for author in authors:
            if not author:
                continue
            parts = author.strip().split()
            if parts:
                last_names.append(parts[-1].lower())
        return last_names
    
    @staticmethod
    def extract_year(text: str) -> str:
        """Extract year from text"""
        if not text:
            return ""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return match.group(0) if match else ""
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
        """Ratio of overlapping tokens"""
        if not tokens1 or not tokens2:
            return 0.0
        set1, set2 = set(tokens1), set(tokens2)
        overlap = len(set1 & set2)
        return overlap / min(len(set1), len(set2))
    
    @staticmethod
    def extract_features(bib: Dict, ref: Dict) -> Dict[str, float]:
        """Extract features for a (BibEntry, RefEntry) pair"""
        features = {}
        
        # Clean texts
        bib_title = FeatureProcessor.clean_title(bib.get('title', ''))
        ref_title = FeatureProcessor.clean_title(ref.get('title', ''))
        
        bib_title_tokens = FeatureProcessor.tokenize(bib_title)
        ref_title_tokens = FeatureProcessor.tokenize(ref_title)
        
        # Feature 1: Title Jaccard Similarity
        features['title_jaccard'] = FeatureProcessor.jaccard_similarity(
            set(bib_title_tokens), set(ref_title_tokens)
        )
        
        # Feature 2: Title Token Overlap Ratio
        features['title_overlap'] = FeatureProcessor.token_overlap_ratio(
            bib_title_tokens, ref_title_tokens
        )
        
        # Feature 3: Normalized Edit Distance using python-Levenshtein (fast!)
        if bib_title and ref_title:
            # Levenshtein.distance is much faster than pure Python
            dist = Levenshtein.distance(bib_title, ref_title)
            features['title_edit_dist'] = 1.0 - (dist / max(len(bib_title), len(ref_title)))
        else:
            features['title_edit_dist'] = 0.0
        
        # Feature 4: Author Last Name Overlap
        bib_authors = FeatureProcessor.extract_author_last_names(bib.get('authors', []))
        ref_authors = FeatureProcessor.extract_author_last_names(ref.get('authors', []))
        features['author_overlap'] = FeatureProcessor.token_overlap_ratio(
            bib_authors, ref_authors
        )
        
        # Feature 5: First Author Match
        features['first_author_match'] = 1.0 if (
            bib_authors and ref_authors and bib_authors[0] == ref_authors[0]
        ) else 0.0
        
        # Feature 6: Year Match
        bib_year = bib.get('year') or FeatureProcessor.extract_year(bib.get('raw_content', ''))
        ref_year = ref.get('year', '')
        features['year_match'] = 1.0 if bib_year == ref_year else 0.0
        
        # Year difference
        try:
            if bib_year and ref_year:
                features['year_diff'] = abs(int(bib_year) - int(ref_year))
            else:
                features['year_diff'] = 10
        except ValueError:
            features['year_diff'] = 10
        
        # Feature 7: ArXiv ID Exact Match
        bib_arxiv = (bib.get('arxiv_id') or '').replace('.', '-')
        ref_arxiv = (ref.get('arxiv_id') or '').replace('.', '-')
        features['arxiv_match'] = 1.0 if (bib_arxiv and ref_arxiv and bib_arxiv == ref_arxiv) else 0.0
        
        # Feature 8: ArXiv ID in raw content
        raw_content = bib.get('raw_content', '')
        ref_arxiv_dot = ref_arxiv.replace('-', '.')
        features['arxiv_in_content'] = 1.0 if ref_arxiv_dot and ref_arxiv_dot in raw_content else 0.0
        
        # Feature 9: Number of matching authors
        features['num_matching_authors'] = len(set(bib_authors) & set(ref_authors))
        
        # Feature 10: Title length ratio
        len_ratio = len(bib_title) / len(ref_title) if ref_title else 0
        features['title_len_ratio'] = min(len_ratio, 1/len_ratio) if len_ratio > 0 else 0
        
        # Feature 11: Combined score
        features['combined_score'] = (
            0.4 * features['title_jaccard'] +
            0.3 * features['author_overlap'] +
            0.2 * features['year_match'] +
            0.1 * features['first_author_match']
        )
        
        return features


def generate_candidate_pairs_worker(pub_data: Dict, max_candidates: int = 50) -> List[Dict]:
    """
    Worker function for multiprocessing.
    Generate candidate pairs for a single publication.
    """
    pairs = []
    
    for bib in pub_data['bibs']:
        # Extract features for all references
        candidates = []
        for arxiv_id, ref in pub_data['refs'].items():
            t1 = bib['title'].replace('\n', ' ').strip()
            t2 = ref['title'].replace('\n', ' ').strip()

            # Quick filter: skip if title lengths are very different
            if abs(len(t1) - len(t2)) > 40:
                continue

            features = FeatureProcessor.extract_features(bib, ref)
            candidates.append({
                'pub_id': pub_data['pub_id'],
                'bib_key': bib['key'],
                'arxiv_id': arxiv_id,
                'features': features
            })
        
        # Sort by combined score and take top candidates
        candidates.sort(key=lambda x: x['features']['combined_score'], reverse=True)
        pairs.extend(candidates[:max_candidates])
    
    return pairs

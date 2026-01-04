"""Feature extraction processor for multiprocessing"""

import re
from typing import Dict, List, Set
import Levenshtein


class FeatureProcessor:
    """Feature extraction for parallel processing"""
    
    FEATURE_NAMES = [
        'title_jaccard', 'title_overlap', 'title_edit_dist',
        'author_overlap', 'first_author_match', 'year_match', 'year_diff',
        'arxiv_match', 'arxiv_in_content', 'num_matching_authors', 
        'title_len_ratio', 'combined_score'
    ]
    
    @staticmethod
    def clean_title(title: str) -> str:
        if not title:
            return ""
        title = re.sub(r'[^\w\s]', ' ', title.lower())
        return re.sub(r'\s+', ' ', title).strip()
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()
    
    @staticmethod
    def extract_author_last_names(authors: List[str], max_authors: int = 50) -> List[str]:
        last_names = []
        for author in authors[:max_authors]:
            if not author or len(author) < 2:
                continue
            author = re.sub(r'[{}\\]', '', author).strip()
            if author:
                parts = author.split()
                if parts and len(parts[-1]) > 1:
                    last_names.append(parts[-1].lower())
        return last_names
    
    @staticmethod
    def extract_year(text: str) -> str:
        if not text:
            return ""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return match.group(0) if match else ""
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    @staticmethod
    def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
        if not tokens1 or not tokens2:
            return 0.0
        set1, set2 = set(tokens1), set(tokens2)
        return len(set1 & set2) / min(len(set1), len(set2))
    
    @staticmethod
    def extract_features(bib: Dict, ref: Dict) -> Dict[str, float]:
        """Extract all features for a (bib, ref) pair"""
        features = {}
        
        # Title features
        bib_title = FeatureProcessor.clean_title(bib.get('title', ''))
        ref_title = FeatureProcessor.clean_title(ref.get('title', ''))
        bib_tokens = FeatureProcessor.tokenize(bib_title)
        ref_tokens = FeatureProcessor.tokenize(ref_title)
        
        features['title_jaccard'] = FeatureProcessor.jaccard_similarity(set(bib_tokens), set(ref_tokens))
        features['title_overlap'] = FeatureProcessor.token_overlap_ratio(bib_tokens, ref_tokens)
        
        # Edit distance (using python-Levenshtein for speed)
        if bib_title and ref_title:
            dist = Levenshtein.distance(bib_title, ref_title)
            features['title_edit_dist'] = 1.0 - (dist / max(len(bib_title), len(ref_title)))
        else:
            features['title_edit_dist'] = 0.0
        
        # Author features
        bib_authors = FeatureProcessor.extract_author_last_names(bib.get('authors', []))
        ref_authors = FeatureProcessor.extract_author_last_names(ref.get('authors', []))
        
        features['author_overlap'] = FeatureProcessor.token_overlap_ratio(bib_authors, ref_authors)
        features['first_author_match'] = 1.0 if (
            bib_authors and ref_authors and bib_authors[0] == ref_authors[0]
        ) else 0.0
        features['num_matching_authors'] = min(len(set(bib_authors) & set(ref_authors)), 20)
        
        # Year features
        bib_year = bib.get('year') or FeatureProcessor.extract_year(bib.get('raw_content', ''))
        ref_year = ref.get('year', '') or (ref.get('submission_date', '')[:4] if ref.get('submission_date') else '')
        
        features['year_match'] = 1.0 if (bib_year and ref_year and bib_year == ref_year) else 0.0
        try:
            features['year_diff'] = min(abs(int(bib_year) - int(ref_year)), 50) if bib_year and ref_year else 10
        except ValueError:
            features['year_diff'] = 10
        
        # ArXiv features
        bib_arxiv = (bib.get('arxiv_id') or '').replace('.', '-')
        ref_arxiv = (ref.get('arxiv_id') or '').replace('.', '-')
        features['arxiv_match'] = 1.0 if (bib_arxiv and ref_arxiv and bib_arxiv == ref_arxiv) else 0.0
        features['arxiv_in_content'] = 1.0 if ref_arxiv.replace('-', '.') in bib.get('raw_content', '') else 0.0
        
        # Title length ratio
        len_ratio = len(bib_title) / len(ref_title) if ref_title else 0
        features['title_len_ratio'] = min(len_ratio, 1/len_ratio) if len_ratio > 0 else 0
        
        # Combined score
        features['combined_score'] = (
            0.4 * features['title_jaccard'] + 0.3 * features['author_overlap'] +
            0.2 * features['year_match'] + 0.1 * features['first_author_match']
        )
        
        return features


def generate_candidate_pairs_worker(pub_data: Dict, max_candidates: int = 50) -> List[Dict]:
    """Worker function for multiprocessing - generates candidate pairs for one publication"""
    pairs = []
    
    for bib in pub_data['bibs']:
        candidates = []
        for arxiv_id, ref in pub_data['refs'].items():
            t1 = bib['title'].replace('\n', ' ').strip()
            t2 = ref['title'].replace('\n', ' ').strip()
            
            if abs(len(t1) - len(t2)) > 40:
                continue
            
            features = FeatureProcessor.extract_features(bib, ref)
            candidates.append({
                'pub_id': pub_data['pub_id'],
                'bib_key': bib['key'],
                'arxiv_id': arxiv_id,
                'features': features
            })
        
        candidates.sort(key=lambda x: x['features']['combined_score'], reverse=True)
        pairs.extend(candidates[:max_candidates])
    
    return pairs

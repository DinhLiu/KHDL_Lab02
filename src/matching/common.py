"""Common utilities for reference matching pipeline

Shared classes:
- TextCleaner: Text preprocessing utilities  
- FeatureExtractor: Feature extraction for reference matching
- ReferenceMatchingModel: Logistic regression model

Configuration:
- MANUAL_PUB_IDS: Publications with manual labels (auto-detected)
- get_labeled_publications(): Auto-detect publications with pred.json
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

# Base paths
BASE_PATH = Path(__file__).parent.parent.parent  # KHDL_Lab02_v2
DATA_DIR = BASE_PATH / "23120260"
OUTPUT_DIR = BASE_PATH / "output"

# Manual partition assignments
MANUAL_PARTITIONS = {
    "2411-00222": "test",
    "2411-00223": "valid",
    "2411-00225": "train",
    "2411-00226": "train",
    "2411-00227": "train",
}


def get_labeled_publications(data_dir: Path = None) -> Dict[str, str]:
    """Auto-detect publications with pred.json files and return their partitions.
    
    Returns:
        Dict mapping pub_id -> partition ('train', 'valid', 'test')
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    partitions = {}
    
    for pub_path in data_dir.iterdir():
        if not pub_path.is_dir():
            continue
        
        pred_file = pub_path / "pred.json"
        if pred_file.exists():
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                # Get partition from pred.json or fallback to MANUAL_PARTITIONS
                partition = pred_data.get('partition', MANUAL_PARTITIONS.get(pub_path.name, 'train'))
                partitions[pub_path.name] = partition
            except (json.JSONDecodeError, IOError):
                partitions[pub_path.name] = MANUAL_PARTITIONS.get(pub_path.name, 'train')
    
    return partitions


def load_ground_truth(data_dir: Path = None) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Load ground truth labels from pred.json files.
    
    Returns:
        Tuple of (ground_truth_lookup, partition_lookup) where:
        - ground_truth_lookup: {pub_id: {bib_key: arxiv_id}}
        - partition_lookup: {pub_id: partition}
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    ground_truth = {}
    partitions = {}
    
    for pub_path in data_dir.iterdir():
        if not pub_path.is_dir():
            continue
        
        pred_file = pub_path / "pred.json"
        if pred_file.exists():
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_data = json.load(f)
                
                pub_id = pub_path.name
                partitions[pub_id] = pred_data.get('partition', 'train')
                
                gt = pred_data.get('groundtruth', {})
                if gt:
                    ground_truth[pub_id] = gt
            except (json.JSONDecodeError, IOError):
                continue
    
    return ground_truth, partitions


# =============================================================================
# TextCleaner
# =============================================================================

class TextCleaner:
    """Text preprocessing utilities for reference matching"""
    
    STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                  'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                  'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
                  'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its'}
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text: lowercase, remove LaTeX commands, normalize whitespace"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \cmd{arg} -> arg
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove remaining LaTeX commands
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean title: remove common prefixes"""
        title = TextCleaner.clean_text(title)
        return re.sub(r'^(on|the|a|an)\s+', '', title)
    
    @staticmethod
    def clean_author_name(name: str) -> str:
        """Extract last name from author string"""
        if not name:
            return ""
        name = TextCleaner.clean_text(name)
        name = re.sub(r'\b(dr|prof|mr|mrs|ms|jr|sr|phd|md)\b', '', name)
        parts = name.split()
        return parts[-1] if parts else name
    
    @staticmethod
    def extract_author_last_names(authors: List[str]) -> List[str]:
        """Extract last names from list of authors"""
        return [TextCleaner.clean_author_name(a) for a in authors if a]
    
    @staticmethod
    def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text, optionally removing stopwords"""
        text = TextCleaner.clean_text(text)
        tokens = text.split()
        return [t for t in tokens if t not in TextCleaner.STOP_WORDS] if remove_stopwords else tokens
    
    @staticmethod
    def extract_year(text: str) -> Optional[str]:
        """Extract 4-digit year from text"""
        if not text:
            return None
        match = re.search(r'\b(19|20)\d{2}\b', str(text))
        return match.group(0) if match else None
    
    @staticmethod
    def extract_arxiv_id(text: str) -> Optional[str]:
        """Extract arXiv ID from text"""
        if not text:
            return None
        for pattern in [r'arxiv[:\s]*(\d{4}[.-]\d{4,5})', r'\b(\d{4}\.\d{4,5})\b']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                arxiv_id = match.group(1).replace('.', '-')
                return arxiv_id if '-' in arxiv_id else arxiv_id[:4] + '-' + arxiv_id[4:]
        return None


# =============================================================================
# FeatureExtractor  
# =============================================================================

class FeatureExtractor:
    """Feature extraction for reference matching (ranking problem)"""
    
    FEATURE_NAMES = [
        'title_jaccard', 'title_overlap', 'title_edit_dist',
        'author_overlap', 'first_author_match', 'year_match', 'year_diff',
        'arxiv_match', 'arxiv_in_content', 'num_matching_authors', 
        'title_len_ratio', 'combined_score'
    ]
    
    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        """Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    @staticmethod
    def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
        """Overlap ratio between two token lists"""
        if not tokens1 or not tokens2:
            return 0.0
        set1, set2 = set(tokens1), set(tokens2)
        return len(set1 & set2) / min(len(set1), len(set2))
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance"""
        if len(s1) < len(s2):
            return FeatureExtractor.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
            prev_row = curr_row
        return prev_row[-1]
    
    @staticmethod
    def extract_features(bib: Dict, ref: Dict) -> Dict[str, float]:
        """Extract all features for a (bib, ref) pair"""
        features = {}
        
        # Title features
        bib_title = TextCleaner.clean_title(bib.get('title', ''))
        ref_title = TextCleaner.clean_title(ref.get('title', ''))
        bib_tokens = TextCleaner.tokenize(bib_title)
        ref_tokens = TextCleaner.tokenize(ref_title)
        
        features['title_jaccard'] = FeatureExtractor.jaccard_similarity(set(bib_tokens), set(ref_tokens))
        features['title_overlap'] = FeatureExtractor.token_overlap_ratio(bib_tokens, ref_tokens)
        
        if bib_title and ref_title:
            dist = FeatureExtractor.levenshtein_distance(bib_title, ref_title)
            features['title_edit_dist'] = 1.0 - dist / max(len(bib_title), len(ref_title))
        else:
            features['title_edit_dist'] = 0.0
        
        # Author features
        bib_authors = TextCleaner.extract_author_last_names(bib.get('authors', [])[:50])
        ref_authors = TextCleaner.extract_author_last_names(ref.get('authors', [])[:50])
        features['author_overlap'] = FeatureExtractor.token_overlap_ratio(bib_authors, ref_authors)
        features['first_author_match'] = 1.0 if bib_authors and ref_authors and bib_authors[0] == ref_authors[0] else 0.0
        features['num_matching_authors'] = min(len(set(bib_authors) & set(ref_authors)), 20)
        
        # Year features
        bib_year = bib.get('year') or TextCleaner.extract_year(bib.get('raw_content', ''))
        ref_year = ref.get('year', '')
        features['year_match'] = 1.0 if bib_year and ref_year and bib_year == ref_year else 0.0
        try:
            features['year_diff'] = min(abs(int(bib_year) - int(ref_year)), 50) if bib_year and ref_year else 10
        except ValueError:
            features['year_diff'] = 10
        
        # ArXiv features
        bib_arxiv = (bib.get('arxiv_id') or '').replace('.', '-')
        ref_arxiv = (ref.get('arxiv_id') or '').replace('.', '-')
        features['arxiv_match'] = 1.0 if bib_arxiv and ref_arxiv and bib_arxiv == ref_arxiv else 0.0
        features['arxiv_in_content'] = 1.0 if ref_arxiv.replace('-', '.') in bib.get('raw_content', '') else 0.0
        
        # Title length ratio
        len_ratio = len(bib_title) / len(ref_title) if ref_title else 0
        features['title_len_ratio'] = min(len_ratio, 1/len_ratio) if len_ratio > 0 else 0
        
        # Combined score
        features['combined_score'] = (0.4 * features['title_jaccard'] + 0.3 * features['author_overlap'] +
                                      0.2 * features['year_match'] + 0.1 * features['first_author_match'])
        return features
    
    @staticmethod
    def features_to_vector(features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array"""
        return np.array([features.get(name, 0.0) for name in FeatureExtractor.FEATURE_NAMES])
    
    @staticmethod
    def extract_features_vector(bib: Dict, ref: Dict) -> np.ndarray:
        """Extract features and return as numpy array"""
        return FeatureExtractor.features_to_vector(FeatureExtractor.extract_features(bib, ref))


# =============================================================================
# ReferenceMatchingModel
# =============================================================================

class ReferenceMatchingModel:
    """Logistic regression model for reference matching"""
    
    FEATURE_NAMES = FeatureExtractor.FEATURE_NAMES
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000, verbose: bool = True):
        """Train model using gradient descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for epoch in range(epochs):
            pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            error = pred - y
            self.weights -= lr * np.dot(X.T, error) / n_samples
            self.bias -= lr * np.sum(error) / n_samples
            
            if verbose and epoch % 200 == 0:
                loss = -np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log(1 - pred + 1e-10))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of match"""
        if self.weights is None:
            # Default weights if not trained
            self.weights = np.array([0.3, 0.2, 0.15, 0.25, 0.1, 0.15, -0.05, 1.0, 0.8, 0.1, 0.05, 0.0])
        return self._sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict match (binary)"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (absolute weights)"""
        if self.weights is None:
            return {}
        return {name: abs(w) for name, w in zip(self.FEATURE_NAMES, self.weights)}
    
    def save(self, path: Path):
        """Save model to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'weights': self.weights.tolist() if self.weights is not None else None,
                'bias': self.bias,
                'feature_names': self.FEATURE_NAMES
            }, f, indent=2)
    
    def load(self, path: Path):
        """Load model from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.weights = np.array(data['weights']) if data['weights'] else None
        self.bias = data['bias']
    
    def rank_candidates(self, bib: Dict, refs: Dict[str, Dict], top_k: int = 5) -> List[Tuple[str, float]]:
        """Rank candidate references for a bibitem"""
        scores = []
        for arxiv_id, ref in refs.items():
            features = FeatureExtractor.extract_features_vector(bib, ref).reshape(1, -1)
            score = self.predict_proba(features)[0]
            scores.append((arxiv_id, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]


# =============================================================================
# Evaluation Metrics
# =============================================================================

def calculate_mrr(predictions: List[List[str]], ground_truth: List[str], top_k: int = 5) -> float:
    """Calculate Mean Reciprocal Rank"""
    rr = []
    for pred_list, true_id in zip(predictions, ground_truth):
        rank = next((i + 1 for i, p in enumerate(pred_list[:top_k]) if p == true_id), 0)
        rr.append(1.0 / rank if rank > 0 else 0.0)
    return np.mean(rr) if rr else 0.0


def calculate_precision_at_k(predictions: List[List[str]], ground_truth: List[str], k: int = 1) -> float:
    """Calculate Precision@K"""
    if not predictions:
        return 0.0
    return sum(1 for pred, true in zip(predictions, ground_truth) if true in pred[:k]) / len(predictions)


# =============================================================================
# Multiprocessing Worker Functions
# =============================================================================

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
            
            features = FeatureExtractor.extract_features(bib, ref)
            candidates.append({
                'pub_id': pub_data['pub_id'],
                'bib_key': bib['key'],
                'arxiv_id': arxiv_id,
                'features': features
            })
        
        candidates.sort(key=lambda x: x['features']['combined_score'], reverse=True)
        pairs.extend(candidates[:max_candidates])
    
    return pairs


# Alias for backward compatibility
FeatureProcessor = FeatureExtractor

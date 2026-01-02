"""
Reference Matching Module

Components for the reference matching notebook pipeline:
- data_structures: BibEntry, RefEntry dataclasses
- feature_extractor: Feature extraction for matching
- text_cleaner: Text cleaning utilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re


@dataclass
class BibEntry:
    """Represents a BibTeX entry extracted from LaTeX"""
    key: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: str = ""
    venue: str = ""
    arxiv_id: str = ""
    raw_content: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue,
            'arxiv_id': self.arxiv_id,
            'raw_content': self.raw_content
        }


@dataclass
class RefEntry:
    """Represents a reference entry from references.json"""
    arxiv_id: str
    title: str = ""
    authors: List[str] = field(default_factory=list)
    submission_date: str = ""
    venue: str = ""
    
    @property
    def year(self) -> str:
        if self.submission_date:
            return self.submission_date[:4]
        return ""
    
    def to_dict(self) -> Dict:
        return {
            'arxiv_id': self.arxiv_id,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'venue': self.venue
        }


@dataclass
class LabeledPair:
    """Represents a labeled pair for training/evaluation"""
    pub_id: str
    bib_key: str
    arxiv_id: str
    is_match: bool
    label_source: str  # 'manual' or 'auto'
    confidence: float = 1.0
    reason: str = ""


class TextCleaner:
    """Text preprocessing for reference matching"""
    
    STOP_WORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'that', 'which', 'who', 'whom', 'this', 'these', 'those', 'it', 'its'
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize title"""
        title = TextCleaner.clean_text(title)
        title = re.sub(r'^(on|the|a|an)\s+', '', title)
        return title
    
    @staticmethod
    def clean_author_name(name: str) -> str:
        """Normalize author name"""
        if not name:
            return ""
        name = TextCleaner.clean_text(name)
        name = re.sub(r'\b(dr|prof|mr|mrs|ms|jr|sr|phd|md)\b', '', name)
        parts = name.split()
        return parts[-1] if parts else name
    
    @staticmethod
    def extract_author_last_names(authors: List[str]) -> List[str]:
        """Extract last names from author list"""
        return [TextCleaner.clean_author_name(a) for a in authors if a]
    
    @staticmethod
    def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text into words"""
        text = TextCleaner.clean_text(text)
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in TextCleaner.STOP_WORDS]
        return tokens
    
    @staticmethod
    def extract_year(text: str) -> Optional[str]:
        """Extract year from text"""
        if not text:
            return None
        match = re.search(r'\b(19|20)\d{2}\b', str(text))
        return match.group(0) if match else None
    
    @staticmethod
    def extract_arxiv_id(text: str) -> Optional[str]:
        """Extract arXiv ID from text"""
        if not text:
            return None
        patterns = [
            r'arxiv[:\s]*(\d{4}[.-]\d{4,5})',
            r'arXiv[:\s]*(\d{4}[.-]\d{4,5})',
            r'\b(\d{4}\.\d{4,5})\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                arxiv_id = match.group(1).replace('.', '-')
                if '-' not in arxiv_id:
                    arxiv_id = arxiv_id[:4] + '-' + arxiv_id[4:]
                return arxiv_id
        return None

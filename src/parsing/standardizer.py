"""
Text Standardization and Normalization

Normalizes LaTeX content through various operations:
- Remove comments
- Normalize whitespace
- Normalize math expressions
- Remove formatting-only commands
- Normalize citations
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class StandardizationConfig:
    """Configuration for standardization operations"""
    normalize_math: bool = True
    remove_comments: bool = True
    normalize_whitespace: bool = True
    remove_formatting: bool = True
    normalize_citations: bool = True


class Standardizer:
    """
    Standardizes LaTeX content through various normalizations.
    
    Example usage:
        standardizer = Standardizer()
        clean_content = standardizer.standardize(raw_latex)
    """
    
    # LaTeX commands to remove (formatting-only, no semantic meaning)
    FORMATTING_COMMANDS = [
        r'\\centering\b',
        r'\\raggedright\b',
        r'\\raggedleft\b',
        r'\[htpb!?\]',
        r'\[h!?\]',
        r'\[t!?\]',
        r'\[b!?\]',
        r'\[p!?\]',
        r'\[H\]',
        r'\\midrule\b',
        r'\\toprule\b',
        r'\\bottomrule\b',
        r'\\hline\b',
        r'\\cline\{[^}]*\}',
        r'\\noindent\b',
        r'\\smallskip\b',
        r'\\medskip\b',
        r'\\bigskip\b',
        r'\\vspace\*?\{[^}]*\}',
        r'\\hspace\*?\{[^}]*\}',
        r'\\vfill\b',
        r'\\hfill\b',
        r'\\newpage\b',
        r'\\clearpage\b',
        r'\\pagebreak\b',
        r'\\linebreak\b',
        r'\\\\(?:\[[^\]]*\])?',
        r'\\par\b',
        r'\\indent\b',
    ]
    
    # Font commands to simplify (pattern, replacement)
    FONT_COMMANDS = [
        (r'\\textbf\{([^}]*)\}', r'\1'),
        (r'\\textit\{([^}]*)\}', r'\1'),
        (r'\\emph\{([^}]*)\}', r'\1'),
        (r'\\underline\{([^}]*)\}', r'\1'),
        (r'\\texttt\{([^}]*)\}', r'\1'),
        (r'\\textrm\{([^}]*)\}', r'\1'),
        (r'\\textsf\{([^}]*)\}', r'\1'),
        (r'\\textsc\{([^}]*)\}', r'\1'),
        (r'\{\\bf\s+([^}]*)\}', r'\1'),
        (r'\{\\it\s+([^}]*)\}', r'\1'),
        (r'\{\\em\s+([^}]*)\}', r'\1'),
        (r'\{\\tt\s+([^}]*)\}', r'\1'),
        (r'\\bfseries\b', ''),
        (r'\\itshape\b', ''),
        (r'\\rmfamily\b', ''),
        (r'\\sffamily\b', ''),
        (r'\\ttfamily\b', ''),
    ]
    
    # Size commands to remove
    SIZE_COMMANDS = [
        r'\\tiny\b',
        r'\\scriptsize\b',
        r'\\footnotesize\b',
        r'\\small\b',
        r'\\normalsize\b',
        r'\\large\b',
        r'\\Large\b',
        r'\\LARGE\b',
        r'\\huge\b',
        r'\\Huge\b',
    ]
    
    def __init__(self, config: StandardizationConfig = None):
        self.config = config or StandardizationConfig()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.formatting_pattern = re.compile(
            '|'.join(self.FORMATTING_COMMANDS)
        )
        self.size_pattern = re.compile(
            '|'.join(self.SIZE_COMMANDS)
        )
    
    def standardize(self, content: str) -> str:
        """
        Apply all standardization operations.
        
        Args:
            content: Raw LaTeX content
            
        Returns:
            Standardized content
        """
        if self.config.remove_comments:
            content = self.remove_comments(content)
        
        if self.config.remove_formatting:
            content = self.remove_formatting(content)
        
        if self.config.normalize_math:
            content = self.normalize_math(content)
        
        if self.config.normalize_citations:
            content = self.normalize_citations(content)
        
        if self.config.normalize_whitespace:
            content = self.normalize_whitespace(content)
        
        return content
    
    def remove_comments(self, content: str) -> str:
        """Remove LaTeX comments (lines starting with %)"""
        # Remove full-line comments
        content = re.sub(r'^%.*$', '', content, flags=re.MULTILINE)
        # Remove inline comments (but not escaped %)
        content = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)
        return content
    
    def remove_formatting(self, content: str) -> str:
        """Remove formatting-only commands"""
        # Remove formatting commands
        content = self.formatting_pattern.sub('', content)
        
        # Remove size commands
        content = self.size_pattern.sub('', content)
        
        # Simplify font commands (keep content)
        for pattern, replacement in self.FONT_COMMANDS:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def normalize_math(self, content: str) -> str:
        """Normalize math expressions to standard format"""
        # Convert \( \) to $ $
        content = re.sub(r'\\\((.+?)\\\)', r'$\1$', content, flags=re.DOTALL)
        
        # Convert \[ \] to equation environment
        def replace_display(match):
            return f'\\begin{{equation}}\n{match.group(1).strip()}\n\\end{{equation}}'
        content = re.sub(r'\\\[(.+?)\\\]', replace_display, content, flags=re.DOTALL)
        
        # Convert $$ $$ to equation environment
        def replace_double_dollar(match):
            return f'\\begin{{equation}}\n{match.group(1).strip()}\n\\end{{equation}}'
        content = re.sub(r'\$\$(.+?)\$\$', replace_double_dollar, content, flags=re.DOTALL)
        
        return content
    
    def normalize_citations(self, content: str) -> str:
        """Normalize citation commands"""
        # Normalize various cite commands to standard \cite{}
        cite_patterns = [
            (r'\\citep\{([^}]+)\}', r'\\cite{\1}'),
            (r'\\citet\{([^}]+)\}', r'\\cite{\1}'),
            (r'\\citeauthor\{([^}]+)\}', r'\\cite{\1}'),
            (r'\\citeyear\{([^}]+)\}', r'\\cite{\1}'),
        ]
        
        for pattern, replacement in cite_patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        content = re.sub(r'[ \t]+', ' ', content)
        # Replace multiple newlines with double newline
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove trailing whitespace
        content = re.sub(r' +$', '', content, flags=re.MULTILINE)
        return content.strip()


class ContentDeduplicator:
    """
    Deduplicates content elements using content hashing.
    
    Example usage:
        dedup = ContentDeduplicator()
        unique_id = dedup.get_or_create_id(content)
    """
    
    def __init__(self):
        self.content_hashes: Dict[str, str] = {}
        self.counter = 0
    
    def get_hash(self, content: str) -> str:
        """Get MD5 hash of content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
    
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
        element_id = f"{prefix}_{self.counter:04d}_{content_hash}"
        self.content_hashes[content_hash] = element_id
        
        return element_id, True
    
    def is_duplicate(self, content: str) -> bool:
        """Check if content is a duplicate"""
        content_hash = self.get_hash(content)
        return content_hash in self.content_hashes


class TextCleaner:
    """Utility class for text cleaning operations"""
    
    # Common stop words for academic text
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
        # Lowercase
        text = text.lower()
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text into words"""
        text = TextCleaner.clean_text(text)
        tokens = text.split()
        if remove_stopwords:
            tokens = [t for t in tokens if t not in TextCleaner.STOP_WORDS]
        return tokens
    
    @staticmethod
    def extract_year(text: str) -> str:
        """Extract year from text"""
        if not text:
            return ""
        match = re.search(r'\b(19|20)\d{2}\b', str(text))
        return match.group(0) if match else ""
    
    @staticmethod
    def extract_arxiv_id(text: str) -> str:
        """Extract arXiv ID from text"""
        if not text:
            return ""
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
        return ""

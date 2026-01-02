"""
LaTeX Parser for Hierarchical Document Processing

Parses LaTeX sources into structured elements, handling:
- Document structure (sections, subsections, etc.)
- Math environments (inline and display)
- Figures, tables, and algorithms
- Bibliography entries
"""

import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class DocumentElement:
    """Represents a parsed element in the document hierarchy"""
    id: str
    content: str
    element_type: str  # document, section, subsection, paragraph, sentence, formula, figure
    level: int
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class LaTeXParser:
    """
    Parser for LaTeX documents that builds hierarchical structure.
    
    Example usage:
        parser = LaTeXParser()
        elements = parser.parse(latex_content)
        hierarchy = parser.build_hierarchy(elements)
    """
    
    # LaTeX commands to remove (formatting-only, no semantic meaning)
    CLEANUP_PATTERNS = [
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
        r'\\allowbreak\b',
        r'\\nopagebreak\b',
        r'\\enlargethispage\{[^}]*\}',
        r'\\phantom\{[^}]*\}',
        r'\\vphantom\{[^}]*\}',
        r'\\hphantom\{[^}]*\}',
        r'\\strut\b',
        r'\\null\b',
        r'\\relax\b',
        r'\\protect\b',
        r'\\sloppy\b',
        r'\\fussy\b',
    ]
    
    # Section-level commands with their hierarchy levels
    SECTION_COMMANDS = {
        'part': 0,
        'chapter': 1,
        'section': 2,
        'subsection': 3,
        'subsubsection': 4,
        'paragraph': 5,
        'subparagraph': 6,
    }
    
    # Math environment patterns
    BLOCK_MATH_ENVS = [
        'equation', 'equation*', 'align', 'align*', 'gather', 'gather*',
        'multline', 'multline*', 'eqnarray', 'eqnarray*', 'displaymath',
        'split', 'aligned', 'gathered', 'cases', 'array'
    ]
    
    # Reference section identifiers
    REFERENCE_PATTERNS = [
        r'\\section\*?\{(?:References|Bibliography|Works Cited)\}',
        r'\\begin\{thebibliography\}',
    ]
    
    def __init__(self):
        self.elements: Dict[str, DocumentElement] = {}
        self.content_to_id: Dict[str, str] = {}  # For deduplication
        self.element_counter = 0
        
    def generate_id(self, content: str, element_type: str) -> str:
        """Generate unique ID for an element, reusing IDs for duplicate content"""
        # Create a hash of the content for deduplication
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        
        # Check if we've seen this exact content before
        if content_hash in self.content_to_id:
            return self.content_to_id[content_hash]
        
        # Generate new ID
        self.element_counter += 1
        element_id = f"{element_type}_{self.element_counter}_{content_hash}"
        self.content_to_id[content_hash] = element_id
        
        return element_id
    
    def clean_latex(self, text: str) -> str:
        """Remove formatting-only LaTeX commands"""
        cleaned = text
        
        for pattern in self.CLEANUP_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove comments
        cleaned = re.sub(r'(?<!\\)%.*$', '', cleaned, flags=re.MULTILINE)
        
        # Normalize whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        
        return cleaned.strip()
    
    def normalize_inline_math(self, text: str) -> str:
        """Convert all inline math to $...$ format"""
        # Convert \(...\) to $...$
        text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
        
        # Convert \begin{math}...\end{math} to $...$
        text = re.sub(r'\\begin\{math\}(.+?)\\end\{math\}', r'$\1$', text, flags=re.DOTALL)
        
        return text
    
    def normalize_block_math(self, text: str) -> str:
        """Convert all block math to equation environment format"""
        # Convert $$...$$ to equation environment
        def replace_display_math(match):
            content = match.group(1).strip()
            return f'\\begin{{equation}}\n{content}\n\\end{{equation}}'
        
        text = re.sub(r'\$\$(.+?)\$\$', replace_display_math, text, flags=re.DOTALL)
        
        # Convert \[...\] to equation environment
        def replace_bracket_math(match):
            content = match.group(1).strip()
            return f'\\begin{{equation}}\n{content}\n\\end{{equation}}'
        
        text = re.sub(r'\\\[(.+?)\\\]', replace_bracket_math, text, flags=re.DOTALL)
        
        return text
    
    def normalize_math(self, text: str) -> str:
        """Normalize all math expressions"""
        text = self.normalize_inline_math(text)
        text = self.normalize_block_math(text)
        return text
    
    def extract_title(self, text: str) -> Optional[str]:
        """Extract document title"""
        match = re.search(r'\\title\{([^}]*)\}', text, re.DOTALL)
        if match:
            return self.clean_latex(match.group(1))
        return None
    
    def extract_abstract(self, text: str) -> Optional[str]:
        """Extract document abstract"""
        match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.DOTALL)
        if match:
            return self.clean_latex(match.group(1))
        return None
    
    def extract_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Extract all sections with their titles and levels.
        
        Returns:
            List of tuples: (section_type, title, level)
        """
        sections = []
        
        # Pattern for all section commands
        pattern = r'\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]*)\}'
        
        for match in re.finditer(pattern, text):
            section_type = match.group(1)
            title = self.clean_latex(match.group(2))
            level = self.SECTION_COMMANDS.get(section_type, 2)
            sections.append((section_type, title, level))
        
        return sections
    
    def extract_bibitems(self, text: str) -> List[Dict]:
        """Extract bibliography items from \\bibitem commands and convert to BibTeX format"""
        bibitems = []
        
        # Pattern for \bibitem{key} or \bibitem[label]{key}
        pattern = r'\\bibitem(?:\[([^\]]*)\])?\{([^}]+)\}(.*?)(?=\\bibitem|\Z|\\end\{thebibliography\})'
        
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            label = match.group(1) or ''
            key = match.group(2)
            content = match.group(3).strip()
            
            # Clean the content
            raw_content = self.clean_latex(content)
            raw_content = re.sub(r'\\newblock\s*', ' ', raw_content)
            
            # Parse into BibTeX fields
            bibtex_entry = self._parse_bibitem_to_bibtex(key, raw_content)
            
            bibitems.append({
                'key': key,
                'label': label,
                'raw_content': raw_content,
                'bibtex': bibtex_entry
            })
        
        return bibitems
    
    def _parse_bibitem_to_bibtex(self, key: str, content: str) -> Dict:
        """
        Parse raw bibitem content into BibTeX fields.
        
        Attempts to extract:
        - authors
        - title
        - journal/booktitle
        - year
        - volume, pages
        - doi, arxiv id
        """
        bibtex = {
            'type': 'article',  # Default type
            'key': key,
            'author': '',
            'title': '',
            'year': '',
            'journal': '',
            'volume': '',
            'pages': '',
            'doi': '',
            'arxiv': '',
        }
        
        # Extract year (4 digits starting with 19 or 20)
        year_match = re.search(r'\b(19|20)\d{2}\b', content)
        if year_match:
            bibtex['year'] = year_match.group(0)
        
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,}/[^\s,]+', content)
        if doi_match:
            bibtex['doi'] = doi_match.group(0).rstrip('.')
        
        # Extract arXiv ID
        arxiv_patterns = [
            r'arXiv[:\s]*(\d{4}\.\d{4,5})',
            r'arxiv[:\s]*(\d{4}\.\d{4,5})',
            r'\b(\d{4}\.\d{4,5})\b',
        ]
        for pattern in arxiv_patterns:
            arxiv_match = re.search(pattern, content, re.IGNORECASE)
            if arxiv_match:
                bibtex['arxiv'] = arxiv_match.group(1)
                break
        
        # Extract volume and pages
        vol_match = re.search(r'\b(\d+)[:\s]*(\d+[-–]\d+)\b', content)
        if vol_match:
            bibtex['volume'] = vol_match.group(1)
            bibtex['pages'] = vol_match.group(2).replace('–', '-')
        else:
            # Try just pages
            pages_match = re.search(r'\b(pp?\.\s*)?(\d+[-–]\d+)\b', content)
            if pages_match:
                bibtex['pages'] = pages_match.group(2).replace('–', '-')
        
        # Try to split into author and title
        # Common pattern: Authors, "Title", Journal...
        # Or: Authors. Title. Journal...
        
        # Look for title in quotes or italics
        title_patterns = [
            r'["\u201c]([^"\u201d]+)["\u201d]',  # Quoted title
            r'\\textit\{([^}]+)\}',  # Italic title
            r'\\emph\{([^}]+)\}',  # Emphasized title
        ]
        
        for pattern in title_patterns:
            title_match = re.search(pattern, content)
            if title_match:
                bibtex['title'] = title_match.group(1).strip()
                break
        
        # If no quoted title, try to extract based on punctuation
        if not bibtex['title']:
            # Split by periods and take the first part that looks like a title
            parts = re.split(r'\.\s+', content)
            if len(parts) >= 2:
                # First part is usually authors, second is title
                bibtex['author'] = parts[0].strip()
                bibtex['title'] = parts[1].strip() if len(parts) > 1 else ''
        else:
            # Author is everything before the title
            title_pos = content.find(bibtex['title'])
            if title_pos > 0:
                author_part = content[:title_pos].strip()
                # Clean up author part
                author_part = re.sub(r'[,\s]+$', '', author_part)
                bibtex['author'] = author_part
        
        # Try to identify journal names (common patterns)
        journal_patterns = [
            r'(Physical Review [A-Z]+)',
            r'(Phys\.\s*Rev\.\s*[A-Z]+)',
            r'(Nature\s*\w*)',
            r'(Science\s*\w*)',
            r'(Journal of [^,\.]+)',
            r'(J\.\s*[A-Z][a-z]+\.?\s*[A-Z]?[a-z]*\.?)',
            r'(Proceedings of [^,\.]+)',
            r'(Proc\.\s*[^,\.]+)',
            r'(IEEE [^,\.]+)',
            r'(ACM [^,\.]+)',
            r'(arXiv preprint)',
        ]
        
        for pattern in journal_patterns:
            journal_match = re.search(pattern, content)
            if journal_match:
                bibtex['journal'] = journal_match.group(1)
                break
        
        # Determine entry type
        if 'proceedings' in content.lower() or 'conference' in content.lower():
            bibtex['type'] = 'inproceedings'
        elif 'book' in content.lower() or 'press' in content.lower():
            bibtex['type'] = 'book'
        elif 'arxiv' in content.lower():
            bibtex['type'] = 'misc'
        elif 'thesis' in content.lower():
            bibtex['type'] = 'phdthesis'
        
        return bibtex
    
    def bibitems_to_bibtex_string(self, bibitems: List[Dict]) -> str:
        """Convert list of bibitems to BibTeX format string"""
        entries = []
        
        for item in bibitems:
            if 'bibtex' not in item:
                continue
            
            bib = item['bibtex']
            entry_type = bib.get('type', 'article')
            key = bib.get('key', 'unknown')
            
            fields = []
            for field in ['author', 'title', 'journal', 'year', 'volume', 'pages', 'doi', 'arxiv']:
                if bib.get(field):
                    value = bib[field]
                    # Escape special characters
                    value = value.replace('{', '\\{').replace('}', '\\}')
                    if field == 'arxiv':
                        fields.append(f'  eprint = {{{value}}}')
                    else:
                        fields.append(f'  {field} = {{{value}}}')
            
            if fields:
                entry = f'@{entry_type}{{{key},\n'
                entry += ',\n'.join(fields)
                entry += '\n}'
                entries.append(entry)
        
        return '\n\n'.join(entries)
    
    def extract_figures(self, text: str) -> List[Dict]:
        """Extract figure environments"""
        figures = []
        pattern = r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1)
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]*)\}', content)
            caption = caption_match.group(1) if caption_match else ""
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]*)\}', content)
            label = label_match.group(1) if label_match else ""
            
            figures.append({
                'content': content,
                'caption': self.clean_latex(caption),
                'label': label
            })
        
        return figures
    
    def extract_tables(self, text: str) -> List[Dict]:
        """Extract table environments"""
        tables = []
        pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1)
            
            # Extract caption
            caption_match = re.search(r'\\caption\{([^}]*)\}', content)
            caption = caption_match.group(1) if caption_match else ""
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]*)\}', content)
            label = label_match.group(1) if label_match else ""
            
            tables.append({
                'content': content,
                'caption': self.clean_latex(caption),
                'label': label
            })
        
        return tables
    
    def extract_equations(self, text: str) -> List[Dict]:
        """Extract equation environments"""
        equations = []
        
        for env in self.BLOCK_MATH_ENVS:
            pattern = rf'\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}'
            for match in re.finditer(pattern, text, re.DOTALL):
                content = match.group(1).strip()
                
                # Extract label if present
                label_match = re.search(r'\\label\{([^}]*)\}', content)
                label = label_match.group(1) if label_match else ""
                
                equations.append({
                    'environment': env,
                    'content': content,
                    'label': label
                })
        
        return equations
    
    def parse(self, content: str) -> Dict[str, DocumentElement]:
        """
        Parse LaTeX content into document elements.
        
        Args:
            content: LaTeX content string
            
        Returns:
            Dictionary of element_id -> DocumentElement
        """
        self.elements = {}
        self.content_to_id = {}
        self.element_counter = 0
        
        # Clean and normalize content
        content = self.clean_latex(content)
        content = self.normalize_math(content)
        
        # Create root document element
        root_id = self.generate_id("document", "document")
        root = DocumentElement(
            id=root_id,
            content="",
            element_type="document",
            level=0
        )
        self.elements[root_id] = root
        
        # Extract title
        title = self.extract_title(content)
        if title:
            title_id = self.generate_id(title, "title")
            self.elements[title_id] = DocumentElement(
                id=title_id,
                content=title,
                element_type="title",
                level=1,
                parent=root_id
            )
            root.children.append(title_id)
        
        # Extract abstract
        abstract = self.extract_abstract(content)
        if abstract:
            abstract_id = self.generate_id(abstract, "abstract")
            self.elements[abstract_id] = DocumentElement(
                id=abstract_id,
                content=abstract,
                element_type="abstract",
                level=1,
                parent=root_id
            )
            root.children.append(abstract_id)
        
        # Extract sections
        for section_type, section_title, level in self.extract_sections(content):
            section_id = self.generate_id(section_title, section_type)
            self.elements[section_id] = DocumentElement(
                id=section_id,
                content=section_title,
                element_type=section_type,
                level=level,
                parent=root_id
            )
            root.children.append(section_id)
        
        # Extract figures
        for fig in self.extract_figures(content):
            fig_id = self.generate_id(fig['caption'] or fig['content'][:50], "figure")
            self.elements[fig_id] = DocumentElement(
                id=fig_id,
                content=fig['caption'],
                element_type="figure",
                level=4,
                parent=root_id,
                metadata={'label': fig['label']}
            )
        
        # Extract equations
        for eq in self.extract_equations(content):
            eq_id = self.generate_id(eq['content'][:50], "equation")
            self.elements[eq_id] = DocumentElement(
                id=eq_id,
                content=eq['content'],
                element_type="equation",
                level=4,
                parent=root_id,
                metadata={'environment': eq['environment'], 'label': eq['label']}
            )
        
        return self.elements
    
    def to_json(self) -> Dict:
        """Convert parsed elements to JSON-serializable format"""
        return {
            elem_id: {
                'id': elem.id,
                'content': elem.content,
                'type': elem.element_type,
                'level': elem.level,
                'parent': elem.parent,
                'children': elem.children,
                'metadata': elem.metadata
            }
            for elem_id, elem in self.elements.items()
        }

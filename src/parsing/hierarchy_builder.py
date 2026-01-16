"""
Hierarchy Builder for LaTeX Documents

Builds hierarchical document structure from parsed LaTeX content,
creating a tree structure with proper parent-child relationships.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class HierarchyNode:
    """Represents a node in the document hierarchy"""
    node_id: str
    node_type: str  # 'document', 'section', 'subsection', 'paragraph', 'sentence', 'formula', 'figure', 'table'
    content: str
    level: int
    hierarchy_path: List[int]  # e.g., [1, 2, 1] = section 1, subsection 2, paragraph 1
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class DocumentHierarchy:
    """Complete document hierarchy"""
    paper_id: str
    nodes: Dict[str, HierarchyNode]
    root_id: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (legacy format)"""
        return {
            'paper_id': self.paper_id,
            'root_id': self.root_id,
            'nodes': {
                node_id: {
                    'node_id': node.node_id,
                    'type': node.node_type,
                    'content': node.content,
                    'hierarchy_path': node.hierarchy_path,
                    'level': node.level,
                    'parent': node.parent,
                    'children': node.children,
                    'metadata': node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            'statistics': {
                'total_nodes': len(self.nodes),
                'sections': len([n for n in self.nodes.values() if n.node_type == 'section']),
                'subsections': len([n for n in self.nodes.values() if n.node_type == 'subsection']),
                'paragraphs': len([n for n in self.nodes.values() if n.node_type == 'paragraph']),
                'sentences': len([n for n in self.nodes.values() if n.node_type == 'sentence']),
                'figures': len([n for n in self.nodes.values() if n.node_type == 'figure']),
                'formulas': len([n for n in self.nodes.values() if n.node_type == 'formula']),
                'lists': len([n for n in self.nodes.values() if n.node_type in ('itemize', 'enumerate', 'description')]),
                'items': len([n for n in self.nodes.values() if n.node_type == 'item']),
            }
        }
    
    def to_output_format(self) -> Dict:
        """
        Convert to the required output format:
        {
            "elements": {
                "id-string-1": "Cleaned latex content of element 1",
                "id-string-2": "Cleaned latex content of element 2"
            },
            "hierarchy": {
                "1": {
                    "id-string-2": "id-string-1",  // child: parent
                    "id-string-3": "id-string-2"
                }
            }
        }
        """
        # Build elements dict: id -> content
        elements = {}
        for node_id, node in self.nodes.items():
            elements[node_id] = node.content
        
        # Build hierarchy dict: child_id -> parent_id
        # Version 1 is the main hierarchy
        hierarchy_v1 = {}
        for node_id, node in self.nodes.items():
            if node.parent is not None:
                hierarchy_v1[node_id] = node.parent
        
        return {
            "elements": elements,
            "hierarchy": {
                "1": hierarchy_v1
            }
        }
    
    def save(self, path: Path):
        """Save hierarchy to JSON file (legacy format)"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_output_format(self, path: Path):
        """Save hierarchy in the required output format"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_output_format(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'DocumentHierarchy':
        """Load hierarchy from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = {}
        for node_id, node_data in data['nodes'].items():
            nodes[node_id] = HierarchyNode(
                node_id=node_data['node_id'],
                node_type=node_data['type'],
                content=node_data['content'],
                level=node_data['level'],
                hierarchy_path=node_data['hierarchy_path'],
                children=node_data['children'],
                parent=node_data['parent'],
                metadata=node_data.get('metadata', {})
            )
        
        return cls(
            paper_id=data['paper_id'],
            nodes=nodes,
            root_id=data['root_id']
        )


class HierarchyBuilder:
    """
    Builds hierarchical structure from LaTeX content.
    
    Hierarchy levels:
    - Level 0: Document root
    - Level 1: Sections
    - Level 2: Subsections  
    - Level 3: Subsubsections
    - Level 4: Paragraphs
    - Level 5: Sentences/Elements
    
    Example usage:
        builder = HierarchyBuilder()
        hierarchy = builder.build(latex_content, "paper_123")
        hierarchy.save(Path("output/hierarchy.json"))
    """
    
    # Section patterns with their levels
    SECTION_PATTERNS = [
        (r'\\section\*?\{([^}]*)\}', 1, 'section'),
        (r'\\subsection\*?\{([^}]*)\}', 2, 'subsection'),
        (r'\\subsubsection\*?\{([^}]*)\}', 3, 'subsubsection'),
        (r'\\paragraph\*?\{([^}]*)\}', 4, 'paragraph'),
    ]
    
    # Special environment patterns
    # Note: Tables are treated as 'figure' type per specification
    ENVIRONMENT_PATTERNS = [
        (r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', 'formula'),
        (r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 'formula'),
        (r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}', 'formula'),
        (r'\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}', 'formula'),
        (r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}', 'formula'),
        (r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', 'figure'),
        (r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', 'figure'),  # Tables are figures
        (r'\\begin\{theorem\}(.*?)\\end\{theorem\}', 'theorem'),
        (r'\\begin\{lemma\}(.*?)\\end\{lemma\}', 'lemma'),
        (r'\\begin\{proof\}(.*?)\\end\{proof\}', 'proof'),
        (r'\\begin\{algorithm\}(.*?)\\end\{algorithm\}', 'algorithm'),
    ]
    
    # Block formula patterns ($$...$$ style) - these are leaf nodes
    BLOCK_FORMULA_PATTERN = r'\$\$(.+?)\$\$'
    
    # List environment patterns (handled separately for item extraction)
    LIST_PATTERNS = [
        (r'\\begin\{itemize\}(.*?)\\end\{itemize\}', 'itemize'),
        (r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', 'enumerate'),
        (r'\\begin\{description\}(.*?)\\end\{description\}', 'description'),
    ]
    
    # Reference section patterns to exclude
    REFERENCE_SECTION_PATTERNS = [
        r'references',
        r'bibliography',
        r'works cited',
        r'literatur',
    ]
    
    def __init__(self, pub_id: str = None):
        self.pub_id = pub_id  # Publication ID for element naming
        self.node_counter = 0
        self.nodes: Dict[str, HierarchyNode] = {}
        self.section_counters = [0, 0, 0, 0, 0]  # For hierarchy numbering
        self.content_hashes: Dict[str, str] = {}  # For deduplication: hash -> node_id
        self.in_reference_section = False  # Track if currently in references
    
    def _generate_id(self, node_type: str = "node") -> str:
        """Generate unique node ID with publication ID prefix.
        
        Format: {pub_id}_{type}_{counter:04d}
        Example: 2411-00230_section_0001
        """
        self.node_counter += 1
        if self.pub_id:
            return f"{self.pub_id}_{node_type}_{self.node_counter:04d}"
        return f"{node_type}_{self.node_counter:04d}"
    
    def _reset(self):
        """Reset builder state"""
        self.node_counter = 0
        self.nodes = {}
        self.section_counters = [0, 0, 0, 0, 0]
        self.content_hashes = {}
        self.in_reference_section = False
        # Note: pub_id is NOT reset, it persists across builds
    
    def _increment_counter(self, level: int) -> List[int]:
        """Increment section counter and return hierarchy path"""
        # Reset lower-level counters
        for i in range(level, len(self.section_counters)):
            if i > level:
                self.section_counters[i] = 0
        
        self.section_counters[level - 1] += 1
        return self.section_counters[:level].copy()
    
    def build(self, content: str, paper_id: str) -> DocumentHierarchy:
        """
        Build document hierarchy from LaTeX content.
        
        Hierarchy Structure:
        - Document (root)
          - Title
          - Abstract (with sentences as children)
          - Sections (level 1)
            - Subsections (level 2)
              - Subsubsections (level 3)
                - Paragraphs (level 4)
                  - Leaf nodes: Sentences, Formulas, Figures
                  - List environments (with items as children)
        
        Leaf nodes (smallest elements):
        - Sentences: Text separated by periods
        - Formulas: Block math (equation environments, $$...$$)
        - Figures: figure/table environments (tables treated as figures)
        
        Exclusions: Reference sections are not parsed
        Inclusions: Acknowledgements, Appendices (including \\section*)
        
        Args:
            content: LaTeX content string
            paper_id: Paper identifier (arXiv ID or Semantic Scholar ID)
            
        Returns:
            DocumentHierarchy object
            
        Note: Element IDs will be formatted as {paper_id}_{type}_{counter}
        """
        self._reset()
        self.pub_id = paper_id  # Set publication ID for element naming
        
        # Create root node
        root_id = self._generate_id("document")
        root = HierarchyNode(
            node_id=root_id,
            node_type='document',
            content=paper_id,
            level=0,
            hierarchy_path=[0]
        )
        self.nodes[root_id] = root
        
        # Extract and add title
        self._extract_title(content, root_id)
        
        # Extract and add abstract with sentence children
        self._extract_abstract(content, root_id)
        
        # Parse main content: sections with leaf nodes properly nested
        self._parse_sections_with_leaves(content, root_id)
        
        return DocumentHierarchy(
            paper_id=paper_id,
            nodes=self.nodes,
            root_id=root_id
        )
    
    def _extract_title(self, content: str, parent_id: str):
        """Extract title from content"""
        match = re.search(r'\\title\{([^}]*)\}', content, re.DOTALL)
        if match:
            title = match.group(1).strip()
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            
            title_id = self._generate_id("title")
            title_node = HierarchyNode(
                node_id=title_id,
                node_type='title',
                content=title,
                level=1,
                hierarchy_path=[0, 1],
                parent=parent_id
            )
            self.nodes[title_id] = title_node
            self.nodes[parent_id].children.append(title_id)
    
    def _extract_abstract(self, content: str, parent_id: str):
        """Extract abstract from content"""
        match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            abstract = re.sub(r'\s+', ' ', abstract)
            
            abstract_id = self._generate_id("abstract")
            abstract_node = HierarchyNode(
                node_id=abstract_id,
                node_type='abstract',
                content=abstract,
                level=1,
                hierarchy_path=[0, 2],
                parent=parent_id
            )
            self.nodes[abstract_id] = abstract_node
            self.nodes[parent_id].children.append(abstract_id)
    
    def _is_reference_section(self, title: str) -> bool:
        """Check if section title indicates a reference section"""
        title_lower = title.lower().strip()
        for pattern in self.REFERENCE_SECTION_PATTERNS:
            if pattern in title_lower:
                return True
        return False
    
    def _normalize_for_dedup(self, content: str) -> str:
        """
        Normalize content for deduplication comparison.
        
        Removes minor formatting differences that shouldn't affect identity:
        - Extra whitespace
        - LaTeX formatting commands (bold, italic, etc.)
        - Comments
        """
        normalized = content
        
        # Remove LaTeX comments
        normalized = re.sub(r'(?<!\\)%.*$', '', normalized, flags=re.MULTILINE)
        
        # Remove formatting commands but keep content
        formatting_patterns = [
            (r'\\textbf\{([^}]*)\}', r'\1'),
            (r'\\textit\{([^}]*)\}', r'\1'),
            (r'\\emph\{([^}]*)\}', r'\1'),
            (r'\\underline\{([^}]*)\}', r'\1'),
            (r'\\texttt\{([^}]*)\}', r'\1'),
            (r'\\textrm\{([^}]*)\}', r'\1'),
            (r'\\textsf\{([^}]*)\}', r'\1'),
            (r'\{\\bf\s+([^}]*)\}', r'\1'),
            (r'\{\\it\s+([^}]*)\}', r'\1'),
            (r'\{\\em\s+([^}]*)\}', r'\1'),
        ]
        for pattern, replacement in formatting_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove spacing commands
        spacing_patterns = [
            r'\\centering\b', r'\\raggedright\b', r'\\raggedleft\b',
            r'\\noindent\b', r'\\smallskip\b', r'\\medskip\b', r'\\bigskip\b',
            r'\\vspace\*?\{[^}]*\}', r'\\hspace\*?\{[^}]*\}',
            r'\\\\(?:\[[^\]]*\])?', r'\\par\b',
        ]
        for pattern in spacing_patterns:
            normalized = re.sub(pattern, '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _get_content_hash(self, content: str) -> str:
        """
        Get hash of content for deduplication.
        
        Content is normalized before hashing to ensure that elements
        with identical semantic content (but minor formatting differences)
        are identified as duplicates.
        """
        normalized = self._normalize_for_dedup(content)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:12]
    
    def _get_or_create_node_id(self, content: str, node_type: str) -> Tuple[str, bool]:
        """
        Get existing node ID for duplicate content or create new one.
        
        Uses content hash for deduplication - if content matches exactly
        across versions, returns the same ID.
        
        Returns:
            Tuple of (node_id, is_new)
            
        ID Format: {pub_id}_{type}_{counter:04d}
        Example: 2411-00230_sentence_0001
        """
        content_hash = self._get_content_hash(content)
        
        if content_hash in self.content_hashes:
            return self.content_hashes[content_hash], False
        
        node_id = self._generate_id(node_type)
        self.content_hashes[content_hash] = node_id
        return node_id, True
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling LaTeX-specific cases.
        
        Splits by periods but handles:
        - Abbreviations (e.g., et al., i.e., e.g.)
        - Math expressions (don't split inside $...$)
        - Numbers (e.g., 3.14)
        """
        if not text or len(text.strip()) < 10:
            return []
        
        # Protect math expressions by replacing them temporarily
        math_pattern = r'\$[^$]+\$'
        math_expressions = re.findall(math_pattern, text)
        for i, expr in enumerate(math_expressions):
            text = text.replace(expr, f'__MATH_{i}__', 1)
        
        # Protect common abbreviations
        abbreviations = [
            (r'\bet\s+al\.', '__ETAL__'),
            (r'\bi\.e\.', '__IE__'),
            (r'\be\.g\.', '__EG__'),
            (r'\bvs\.', '__VS__'),
            (r'\bFig\.', '__FIG__'),
            (r'\bEq\.', '__EQ__'),
            (r'\bRef\.', '__REF__'),
            (r'\bSec\.', '__SEC__'),
            (r'\betc\.', '__ETC__'),
            (r'\bDr\.', '__DR__'),
            (r'\bMr\.', '__MR__'),
            (r'\bMs\.', '__MS__'),
            (r'\bProf\.', '__PROF__'),
        ]
        for pattern, replacement in abbreviations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Protect decimal numbers
        text = re.sub(r'(\d)\.(\d)', r'\1__DOT__\2', text)
        
        # Split by sentence-ending punctuation followed by space and capital or end
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
        
        # Restore protected content
        result = []
        for sentence in sentences:
            s = sentence.strip()
            if len(s) < 5:  # Skip very short fragments
                continue
            
            # Restore abbreviations
            s = s.replace('__ETAL__', 'et al.')
            s = s.replace('__IE__', 'i.e.')
            s = s.replace('__EG__', 'e.g.')
            s = s.replace('__VS__', 'vs.')
            s = s.replace('__FIG__', 'Fig.')
            s = s.replace('__EQ__', 'Eq.')
            s = s.replace('__REF__', 'Ref.')
            s = s.replace('__SEC__', 'Sec.')
            s = s.replace('__ETC__', 'etc.')
            s = s.replace('__DR__', 'Dr.')
            s = s.replace('__MR__', 'Mr.')
            s = s.replace('__MS__', 'Ms.')
            s = s.replace('__PROF__', 'Prof.')
            s = s.replace('__DOT__', '.')
            
            # Restore math expressions
            for i, expr in enumerate(math_expressions):
                s = s.replace(f'__MATH_{i}__', expr)
            
            if s:
                result.append(s)
        
        return result
    
    def _parse_sections_with_leaves(self, content: str, root_id: str):
        """
        Parse sections and build hierarchy with proper nesting of leaf nodes.
        
        Leaf nodes (sentences, formulas, figures) are placed under their 
        containing section, not under root.
        
        Exclusions: Reference sections
        Inclusions: Acknowledgements, Appendices (\\section*)
        """
        current_parents = {0: root_id}  # level -> current parent at that level
        
        # Find all section-level commands with their content boundaries
        all_sections = []
        for pattern, level, sec_type in self.SECTION_PATTERNS:
            for match in re.finditer(pattern, content):
                all_sections.append({
                    'position': match.start(),
                    'end_position': match.end(),
                    'title': match.group(1).strip(),
                    'level': level,
                    'type': sec_type
                })
        
        # Sort by position
        all_sections.sort(key=lambda x: x['position'])
        
        # Add end positions for section content extraction
        for i, section in enumerate(all_sections):
            if i + 1 < len(all_sections):
                section['content_end'] = all_sections[i + 1]['position']
            else:
                section['content_end'] = len(content)
        
        # Build hierarchy
        in_reference_section = False
        for section in all_sections:
            title = section['title']
            level = section['level']
            
            # Check if this is a reference section
            if level == 1 and self._is_reference_section(title):
                in_reference_section = True
                continue  # Skip reference sections
            
            # Reset reference flag if we hit a new top-level section
            if level == 1 and not self._is_reference_section(title):
                in_reference_section = False
            
            # Skip if we're inside a reference section
            if in_reference_section:
                continue
            
            # Determine parent (closest ancestor with lower level)
            parent_id = root_id
            for l in range(level - 1, -1, -1):
                if l in current_parents:
                    parent_id = current_parents[l]
                    break
            
            # Create section node (with deduplication)
            section_id, is_new = self._get_or_create_node_id(title, section['type'])
            hierarchy_path = self._increment_counter(level)
            
            if is_new:
                section_node = HierarchyNode(
                    node_id=section_id,
                    node_type=section['type'],
                    content=title,
                    level=level,
                    hierarchy_path=hierarchy_path,
                    parent=parent_id
                )
                self.nodes[section_id] = section_node
            else:
                # Update parent relationship for existing node
                self.nodes[section_id].parent = parent_id
            
            self.nodes[parent_id].children.append(section_id)
            
            # Update current parent at this level
            current_parents[level] = section_id
            
            # Extract leaf nodes from section content (placed under this section)
            section_content = content[section['end_position']:section['content_end']]
            self._extract_leaf_nodes(section_content, section_id, level + 1)
    
    def _extract_leaf_nodes(self, content: str, parent_id: str, level: int):
        """
        Extract all leaf nodes from content and add under parent.
        
        Leaf nodes are:
        - Sentences (text separated by periods)
        - Block Formulas (equation environments, $$...$$)
        - Figures (figure/table environments)
        - List environments (with items as children)
        """
        # Track what we've processed to avoid duplicates
        processed_ranges = []
        
        # 1. Extract figures (including tables as figures)
        figure_patterns = [
            (r'\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}', 'figure'),
            (r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', 'figure'),  # Tables are figures
        ]
        for pattern, fig_type in figure_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                processed_ranges.append((match.start(), match.end()))
                fig_content = match.group(1).strip()
                
                # Extract caption for display
                caption = ""
                caption_match = re.search(r'\\caption\{([^}]*)\}', fig_content)
                if caption_match:
                    caption = caption_match.group(1).strip()
                
                display_content = caption or fig_content[:200]
                fig_id, is_new = self._get_or_create_node_id(display_content, fig_type)
                
                if is_new:
                    fig_node = HierarchyNode(
                        node_id=fig_id,
                        node_type=fig_type,
                        content=display_content,
                        level=level,
                        hierarchy_path=[0],
                        parent=parent_id,
                        metadata={'full_content': fig_content[:500]}
                    )
                    self.nodes[fig_id] = fig_node
                
                self.nodes[parent_id].children.append(fig_id)
        
        # 2. Extract block formulas (equation environments)
        formula_patterns = [
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}',
            r'\\begin\{multline\*?\}(.*?)\\end\{multline\*?\}',
            r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}',
        ]
        for pattern in formula_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                processed_ranges.append((match.start(), match.end()))
                formula_content = match.group(1).strip()
                
                formula_id, is_new = self._get_or_create_node_id(formula_content, 'formula')
                
                if is_new:
                    formula_node = HierarchyNode(
                        node_id=formula_id,
                        node_type='formula',
                        content=formula_content,
                        level=level,
                        hierarchy_path=[0],
                        parent=parent_id
                    )
                    self.nodes[formula_id] = formula_node
                
                self.nodes[parent_id].children.append(formula_id)
        
        # 3. Extract $$...$$ block formulas
        for match in re.finditer(self.BLOCK_FORMULA_PATTERN, content, re.DOTALL):
            processed_ranges.append((match.start(), match.end()))
            formula_content = match.group(1).strip()
            
            formula_id, is_new = self._get_or_create_node_id(formula_content, 'formula')
            
            if is_new:
                formula_node = HierarchyNode(
                    node_id=formula_id,
                    node_type='formula',
                    content=formula_content,
                    level=level,
                    hierarchy_path=[0],
                    parent=parent_id
                )
                self.nodes[formula_id] = formula_node
            
            self.nodes[parent_id].children.append(formula_id)
        
        # 4. Extract list environments (itemize/enumerate) with items as children
        for pattern, list_type in self.LIST_PATTERNS:
            for match in re.finditer(pattern, content, re.DOTALL):
                processed_ranges.append((match.start(), match.end()))
                list_content = match.group(1).strip()
                
                if not list_content or len(list_content) < 5:
                    continue
                
                # Extract items
                items = self._extract_items_from_list(list_content)
                if not items:
                    continue
                
                # Create list parent node
                list_id = self._generate_id(list_type)
                list_node = HierarchyNode(
                    node_id=list_id,
                    node_type=list_type,
                    content=f"{list_type} ({len(items)} items)",
                    level=level,
                    hierarchy_path=[0],
                    parent=parent_id,
                    metadata={'item_count': len(items)}
                )
                self.nodes[list_id] = list_node
                self.nodes[parent_id].children.append(list_id)
                
                # Create child nodes for each item (items are leaf nodes)
                for idx, item_content in enumerate(items):
                    item_id, is_new = self._get_or_create_node_id(item_content, 'item')
                    
                    if is_new:
                        item_node = HierarchyNode(
                            node_id=item_id,
                            node_type='item',
                            content=item_content,
                            level=level + 1,
                            hierarchy_path=[0],
                            parent=list_id,
                            metadata={'item_index': idx + 1}
                        )
                        self.nodes[item_id] = item_node
                    
                    list_node.children.append(item_id)
        
        # 5. Extract sentences from remaining text (excluding processed ranges)
        self._extract_sentences_from_text(content, parent_id, level, processed_ranges)
    
    def _extract_sentences_from_text(self, content: str, parent_id: str, level: int, 
                                     exclude_ranges: List[Tuple[int, int]]):
        """Extract sentences from text, excluding already processed ranges."""
        # Sort ranges and merge overlapping
        exclude_ranges = sorted(exclude_ranges, key=lambda x: x[0])
        
        # Build text from non-excluded ranges
        text_parts = []
        last_end = 0
        for start, end in exclude_ranges:
            if start > last_end:
                text_parts.append(content[last_end:start])
            last_end = max(last_end, end)
        if last_end < len(content):
            text_parts.append(content[last_end:])
        
        clean_content = ' '.join(text_parts)
        
        # Remove remaining environments and commands
        clean_content = re.sub(
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            '', clean_content, flags=re.DOTALL
        )
        
        # Remove section commands
        for pattern, _, _ in self.SECTION_PATTERNS:
            clean_content = re.sub(pattern, '', clean_content)
        
        # Clean up LaTeX commands
        clean_content = re.sub(r'\\label\{[^}]*\}', '', clean_content)
        clean_content = re.sub(r'\\ref\{[^}]*\}', '[ref]', clean_content)
        clean_content = re.sub(r'\\cite[a-z]*\{[^}]*\}', '[cite]', clean_content)
        
        # Normalize whitespace
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        if len(clean_content) < 20:
            return
        
        # Split into sentences
        sentences = self._split_into_sentences(clean_content)
        
        # Add sentence nodes (leaf nodes)
        for sentence in sentences:
            if len(sentence) < 10:
                continue
            
            sentence_id, is_new = self._get_or_create_node_id(sentence, 'sentence')
            
            if is_new:
                sentence_node = HierarchyNode(
                    node_id=sentence_id,
                    node_type='sentence',
                    content=sentence,
                    level=level,
                    hierarchy_path=[0],
                    parent=parent_id
                )
                self.nodes[sentence_id] = sentence_node
            
            self.nodes[parent_id].children.append(sentence_id)
    
    def _extract_items_from_list(self, list_content: str) -> List[str]:
        """
        Extract individual \\item contents from a list environment.
        
        Args:
            list_content: Content inside itemize/enumerate environment
            
        Returns:
            List of item contents
        """
        items = []
        
        # Split by \\item, handling optional arguments like \\item[label]
        # Pattern matches \\item or \\item[...]
        item_pattern = r'\\item(?:\[[^\]]*\])?\s*'
        
        # Find all item positions
        item_matches = list(re.finditer(item_pattern, list_content))
        
        for i, match in enumerate(item_matches):
            start = match.end()
            # End is either next \\item or end of content
            if i + 1 < len(item_matches):
                end = item_matches[i + 1].start()
            else:
                end = len(list_content)
            
            item_content = list_content[start:end].strip()
            
            # Clean up the item content
            item_content = re.sub(r'\s+', ' ', item_content)
            
            if item_content and len(item_content) > 3:
                items.append(item_content)
        
        return items


def build_hierarchy_for_publication(pub_path: Path) -> Optional[DocumentHierarchy]:
    """
    Build hierarchy for a publication.
    
    Args:
        pub_path: Path to publication directory
        
    Returns:
        DocumentHierarchy or None if no content found
    """
    from .file_gatherer import FileGatherer
    
    gatherer = FileGatherer()
    publication = gatherer.gather_publication(pub_path)
    
    if not publication:
        return None
    
    # Get merged content
    merged_content = gatherer.merge_tex_content(publication)
    if not merged_content:
        return None
    
    # Build hierarchy
    builder = HierarchyBuilder()
    return builder.build(merged_content, publication.pub_id)

"""
Hierarchical Parsing and Standardization Module

Components:
- parsing: Main pipeline controller
- file_gatherer: Multi-file gathering from LaTeX sources
- latex_parser: LaTeX document parsing
- standardizer: Text standardization and normalization
- hierarchy_builder: Hierarchical structure construction
- deduplicator: Cross-version deduplication
"""

from .file_gatherer import FileGatherer, TexFile, GatheredPublication
from .latex_parser import LaTeXParser, DocumentElement
from .standardizer import Standardizer, StandardizationConfig
from .hierarchy_builder import HierarchyBuilder, DocumentHierarchy
from .parsing import ParsingPipeline, ParsingResult, parse_publication, parse_all_publications
from .deduplicator import (
    ReferenceDeduplicator, 
    ContentDeduplicator, 
    VersionedHierarchyDeduplicator,
    DeduplicatedReference
)

__all__ = [
    # Main controller
    'ParsingPipeline', 'ParsingResult',
    'parse_publication', 'parse_all_publications',
    # Components
    'FileGatherer', 'TexFile', 'GatheredPublication',
    'LaTeXParser', 'DocumentElement',
    'Standardizer', 'StandardizationConfig',
    'HierarchyBuilder', 'DocumentHierarchy',
    # Deduplication
    'ReferenceDeduplicator', 'ContentDeduplicator', 
    'VersionedHierarchyDeduplicator', 'DeduplicatedReference'
]

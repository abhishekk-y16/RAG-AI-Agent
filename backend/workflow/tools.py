"""
Workflow Tool Implementations - Phase 5
Concrete tools that can be used in workflows
"""

from typing import Dict, Any, List, Optional
import json
from abc import ABC, abstractmethod
from enum import Enum
import re
from datetime import datetime


class ToolResult:
    """Result from a tool execution"""
    
    def __init__(self, success: bool, data: Any = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }


class TextExtractionTool:
    """Extract specific fields from text documents"""
    
    def __init__(self):
        self.patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "date": r"\b(?:0?[1-9]|[12][0-9]|3[01])[-./](?:0?[1-9]|1[0-2])[-./](?:\d{4}|\d{2})\b",
            "amount": r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
            "url": r"https?://[^\s]+",
            "zipcode": r"\b\d{5}(?:-\d{4})?\b"
        }
    
    def execute(self, text: str, fields: List[str]) -> ToolResult:
        """
        Extract fields from text
        
        Args:
            text: Input text
            fields: List of field names to extract (e.g., ['email', 'phone', 'amount'])
        
        Returns:
            ToolResult with extracted fields
        """
        try:
            extracted = {}
            
            for field in fields:
                if field in self.patterns:
                    matches = re.findall(self.patterns[field], text)
                    extracted[field] = matches if matches else None
            
            return ToolResult(
                success=True,
                data={"extracted_fields": extracted, "count": len(extracted)}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class TextValidationTool:
    """Validate text against various criteria"""
    
    VALIDATORS = {
        "email": lambda x: re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", x) is not None,
        "phone": lambda x: re.match(r"^\d{3}[-.]?\d{3}[-.]?\d{4}$", x) is not None,
        "url": lambda x: x.startswith(("http://", "https://")),
        "date": lambda x: re.match(r"^\d{4}-\d{2}-\d{2}$", x) is not None,
        "number": lambda x: re.match(r"^-?\d+\.?\d*$", x) is not None,
        "required": lambda x: bool(x and str(x).strip()),
        "min_length": lambda x, min_len: len(str(x)) >= min_len,
        "max_length": lambda x, max_len: len(str(x)) <= max_len
    }
    
    def execute(self, data: Dict[str, Any], rules: Dict[str, Any]) -> ToolResult:
        """
        Validate data against rules
        
        Args:
            data: Data to validate
            rules: Validation rules (e.g., {"email": "email", "age": "number"})
        
        Returns:
            ToolResult with validation results
        """
        try:
            errors = []
            warnings = []
            
            for field, validation in rules.items():
                if field not in data:
                    errors.append(f"Missing required field: {field}")
                    continue
                
                value = data[field]
                
                # Handle string validation rules
                if isinstance(validation, str):
                    if validation in self.VALIDATORS:
                        validator = self.VALIDATORS[validation]
                        if not validator(value):
                            errors.append(f"Field '{field}' failed '{validation}' validation")
                
                # Handle dict-based rules
                elif isinstance(validation, dict):
                    for rule_type, rule_value in validation.items():
                        if rule_type == "type":
                            if not isinstance(value, rule_value):
                                errors.append(f"Field '{field}' has wrong type")
                        elif rule_type == "enum":
                            if value not in rule_value:
                                errors.append(f"Field '{field}' value not in allowed list")
                        elif rule_type == "min_length":
                            if len(str(value)) < rule_value:
                                errors.append(f"Field '{field}' is too short (min: {rule_value})")
                        elif rule_type == "max_length":
                            if len(str(value)) > rule_value:
                                errors.append(f"Field '{field}' is too long (max: {rule_value})")
            
            valid = len(errors) == 0
            
            return ToolResult(
                success=True,
                data={
                    "valid": valid,
                    "errors": errors,
                    "warnings": warnings,
                    "fields_validated": len(rules)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class TextSummarizationTool:
    """Summarize text content"""
    
    def execute(self, text: str, max_sentences: int = 3) -> ToolResult:
        """
        Create a summary of text
        
        Args:
            text: Text to summarize
            max_sentences: Maximum sentences in summary
        
        Returns:
            ToolResult with summary
        """
        try:
            # Simple extraction-based summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Score sentences by word count (simple heuristic)
            scored = [(i, len(s.split())) for i, s in enumerate(sentences)]
            scored.sort(key=lambda x: x[1], reverse=True)
            
            # Select top sentences in original order
            selected = sorted([s[0] for s in scored[:max_sentences]])
            summary = '. '.join([sentences[i] for i in selected if i < len(sentences)])
            
            return ToolResult(
                success=True,
                data={
                    "summary": summary + ".",
                    "original_length": len(text.split()),
                    "summary_length": len(summary.split()),
                    "compression_ratio": len(summary) / len(text) if text else 0
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DocumentClassificationTool:
    """Classify documents by type or category"""
    
    KEYWORDS = {
        "insurance_claim": ["claim", "policy", "insurance", "coverage", "premium"],
        "contract": ["agreement", "contract", "terms", "conditions", "party"],
        "invoice": ["invoice", "bill", "amount", "payment", "due date"],
        "report": ["report", "analysis", "findings", "conclusion", "recommendation"]
    }
    
    def execute(self, text: str, categories: Optional[List[str]] = None) -> ToolResult:
        """
        Classify document
        
        Args:
            text: Document text
            categories: Categories to check (if None, check all)
        
        Returns:
            ToolResult with classification
        """
        try:
            text_lower = text.lower()
            
            # Use provided categories or default to all
            check_categories = categories if categories else list(self.KEYWORDS.keys())
            
            scores = {}
            for category in check_categories:
                if category in self.KEYWORDS:
                    keywords = self.KEYWORDS[category]
                    matches = sum(1 for kw in keywords if kw in text_lower)
                    score = matches / len(keywords)
                    scores[category] = score
            
            # Find best match
            best_category = max(scores.items(), key=lambda x: x[1]) if scores else None
            
            return ToolResult(
                success=True,
                data={
                    "classification": best_category[0] if best_category else "unknown",
                    "confidence": best_category[1] if best_category else 0,
                    "all_scores": scores
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class DataCleaningTool:
    """Clean and normalize text data"""
    
    def execute(self, data: Dict[str, Any], rules: Dict[str, str]) -> ToolResult:
        """
        Clean data according to rules
        
        Args:
            data: Data to clean
            rules: Cleaning rules (strip, lowercase, uppercase, etc.)
        
        Returns:
            ToolResult with cleaned data
        """
        try:
            cleaned = {}
            
            for field, value in data.items():
                if field not in rules:
                    cleaned[field] = value
                    continue
                
                rule = rules[field]
                text = str(value)
                
                if rule == "strip":
                    text = text.strip()
                elif rule == "lowercase":
                    text = text.lower()
                elif rule == "uppercase":
                    text = text.upper()
                elif rule == "remove_spaces":
                    text = text.replace(" ", "")
                elif rule == "remove_special":
                    text = re.sub(r'[^\w\s]', '', text)
                
                cleaned[field] = text
            
            return ToolResult(
                success=True,
                data={"cleaned_data": cleaned, "fields_cleaned": len(rules)}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ComparisonTool:
    """Compare documents or data"""
    
    def execute(self, text1: str, text2: str) -> ToolResult:
        """
        Compare two documents
        
        Args:
            text1: First document
            text2: Second document
        
        Returns:
            ToolResult with comparison metrics
        """
        try:
            # Simple similarity metrics
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = words1 & words2
            union = words1 | words2
            
            # Jaccard similarity
            jaccard = len(intersection) / len(union) if union else 0
            
            return ToolResult(
                success=True,
                data={
                    "similarity_score": jaccard,
                    "common_words": len(intersection),
                    "unique_in_doc1": len(words1 - words2),
                    "unique_in_doc2": len(words2 - words1)
                }
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ToolFactory:
    """Factory for creating and managing workflow tools"""
    
    AVAILABLE_TOOLS = {
        "extract": TextExtractionTool,
        "validate": TextValidationTool,
        "summarize": TextSummarizationTool,
        "classify": DocumentClassificationTool,
        "clean": DataCleaningTool,
        "compare": ComparisonTool
    }
    
    @classmethod
    def get_tool(cls, tool_name: str):
        """Get a tool instance"""
        if tool_name not in cls.AVAILABLE_TOOLS:
            raise ValueError(f"Unknown tool: {tool_name}")
        return cls.AVAILABLE_TOOLS[tool_name]()
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List available tools"""
        return list(cls.AVAILABLE_TOOLS.keys())
    
    @classmethod
    def get_tool_info(cls) -> Dict[str, Any]:
        """Get information about all tools"""
        return {
            "extract": {
                "description": "Extract specific fields from text documents",
                "params": {"text": "str", "fields": "List[str]"}
            },
            "validate": {
                "description": "Validate data against rules",
                "params": {"data": "Dict", "rules": "Dict"}
            },
            "summarize": {
                "description": "Summarize text content",
                "params": {"text": "str", "max_sentences": "int"}
            },
            "classify": {
                "description": "Classify documents by type",
                "params": {"text": "str", "categories": "List[str]"}
            },
            "clean": {
                "description": "Clean and normalize data",
                "params": {"data": "Dict", "rules": "Dict"}
            },
            "compare": {
                "description": "Compare documents or data",
                "params": {"text1": "str", "text2": "str"}
            }
        }

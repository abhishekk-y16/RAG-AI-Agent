"""
Structured Output Schemas - Phase 5
Defines output schemas for different use cases with JSON Schema validation
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
from jsonschema import validate, ValidationError


class FieldType(str, Enum):
    """Data field types"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATE = "date"
    EMAIL = "email"
    URL = "url"


@dataclass
class SchemaField:
    """Field definition in output schema"""
    name: str
    type: FieldType
    description: str = ""
    required: bool = True
    pattern: Optional[str] = None
    enum: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema property"""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.enum:
            schema["enum"] = self.enum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        
        return schema


@dataclass
class OutputSchema:
    """Complete output schema definition"""
    name: str
    description: str
    fields: List[SchemaField] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema"""
        required_fields = [f.name for f in self.fields if f.required]
        
        schema = {
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                f.name: f.to_json_schema() for f in self.fields
            },
            "required": required_fields
        }
        
        return schema
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate data against schema
        
        Returns:
            (is_valid, error_message)
        """
        try:
            validate(instance=data, schema=self.to_json_schema())
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    def add_field(self, field: SchemaField) -> None:
        """Add a field to the schema"""
        self.fields.append(field)
    
    def add_example(self, example: Dict[str, Any]) -> None:
        """Add an example"""
        self.examples.append(example)


class OutputFormatter:
    """Formats structured output in different formats"""
    
    @staticmethod
    def to_json(data: Dict[str, Any], pretty: bool = True) -> str:
        """Format as JSON"""
        return json.dumps(data, indent=2 if pretty else None)
    
    @staticmethod
    def to_markdown(data: Dict[str, Any], title: str = "") -> str:
        """Format as Markdown"""
        lines = []
        
        if title:
            lines.append(f"# {title}\n")
        
        def format_dict(d: Dict, indent: int = 0) -> None:
            prefix = "  " * indent
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}## {key}")
                    format_dict(value, indent + 1)
                elif isinstance(value, list):
                    lines.append(f"{prefix}### {key}")
                    for item in value:
                        if isinstance(item, dict):
                            format_dict(item, indent + 2)
                        else:
                            lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}- **{key}**: {value}")
            lines.append("")
        
        format_dict(data)
        return "\n".join(lines)
    
    @staticmethod
    def to_table(data: Union[Dict, List[Dict]]) -> str:
        """Format as markdown table"""
        
        # Handle single dict
        if isinstance(data, dict):
            lines = ["| Key | Value |", "|-----|-------|"]
            for key, value in data.items():
                lines.append(f"| {key} | {value} |")
            return "\n".join(lines)
        
        # Handle list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            keys = list(data[0].keys())
            
            # Header
            lines = [f"| {' | '.join(keys)} |"]
            lines.append(f"|{' --- |' * len(keys)}")
            
            # Rows
            for item in data:
                values = [str(item.get(k, "")) for k in keys]
                lines.append(f"| {' | '.join(values)} |")
            
            return "\n".join(lines)
        
        return str(data)
    
    @staticmethod
    def to_text(data: Dict[str, Any], title: str = "") -> str:
        """Format as plain text"""
        lines = []
        
        if title:
            lines.append(f"{title}\n")
            lines.append("=" * len(title) + "\n")
        
        def format_dict(d: Dict, indent: int = 0) -> None:
            prefix = "  " * indent
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append(f"{prefix}{key}:")
                    format_dict(value, indent + 1)
                elif isinstance(value, list):
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            format_dict(item, indent + 2)
                        else:
                            lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}{key}: {value}")
        
        format_dict(data)
        return "\n".join(lines)


# Predefined Output Schemas

def create_extraction_schema() -> OutputSchema:
    """Schema for field extraction results"""
    schema = OutputSchema(
        name="ExtractionResult",
        description="Result from document field extraction"
    )
    
    schema.add_field(SchemaField(
        "extracted_fields", FieldType.OBJECT,
        "Dictionary of extracted fields and values"
    ))
    
    schema.add_field(SchemaField(
        "confidence", FieldType.NUMBER,
        "Overall extraction confidence (0-1)", required=False
    ))
    
    schema.add_field(SchemaField(
        "missing_fields", FieldType.ARRAY,
        "Fields that could not be extracted", required=False
    ))
    
    schema.add_example({
        "extracted_fields": {
            "claim_id": "CLM-123456",
            "date": "2024-01-31",
            "amount": 5000.00
        },
        "confidence": 0.95,
        "missing_fields": []
    })
    
    return schema


def create_qa_schema() -> OutputSchema:
    """Schema for Q&A results"""
    schema = OutputSchema(
        name="QAResult",
        description="Result from question answering"
    )
    
    schema.add_field(SchemaField(
        "question", FieldType.STRING,
        "The original question"
    ))
    
    schema.add_field(SchemaField(
        "answer", FieldType.STRING,
        "Generated answer"
    ))
    
    schema.add_field(SchemaField(
        "sources", FieldType.ARRAY,
        "Source chunks used for answer"
    ))
    
    schema.add_field(SchemaField(
        "confidence", FieldType.NUMBER,
        "Answer confidence score", required=False
    ))
    
    schema.add_example({
        "question": "What is the claim deadline?",
        "answer": "The claim must be submitted within 30 days of the incident.",
        "sources": ["chunk_1", "chunk_2"],
        "confidence": 0.88
    })
    
    return schema


def create_summary_schema() -> OutputSchema:
    """Schema for document summary"""
    schema = OutputSchema(
        name="SummaryResult",
        description="Result from document summarization"
    )
    
    schema.add_field(SchemaField(
        "title", FieldType.STRING,
        "Document title"
    ))
    
    schema.add_field(SchemaField(
        "summary", FieldType.STRING,
        "Document summary"
    ))
    
    schema.add_field(SchemaField(
        "key_points", FieldType.ARRAY,
        "Key points from the document"
    ))
    
    schema.add_field(SchemaField(
        "length", FieldType.INTEGER,
        "Summary length in words", required=False
    ))
    
    schema.add_example({
        "title": "Insurance Claim Form",
        "summary": "Form for submitting insurance claims...",
        "key_points": [
            "Claims must be submitted within 30 days",
            "Documentation required",
            "Supporting evidence needed"
        ],
        "length": 45
    })
    
    return schema


def create_validation_schema() -> OutputSchema:
    """Schema for validation results"""
    schema = OutputSchema(
        name="ValidationResult",
        description="Result from document validation"
    )
    
    schema.add_field(SchemaField(
        "valid", FieldType.BOOLEAN,
        "Whether document passed validation"
    ))
    
    schema.add_field(SchemaField(
        "validation_type", FieldType.STRING,
        "Type of validation performed"
    ))
    
    schema.add_field(SchemaField(
        "errors", FieldType.ARRAY,
        "Validation errors found", required=False
    ))
    
    schema.add_field(SchemaField(
        "warnings", FieldType.ARRAY,
        "Non-critical warnings", required=False
    ))
    
    schema.add_field(SchemaField(
        "confidence", FieldType.NUMBER,
        "Validation confidence", required=False
    ))
    
    schema.add_example({
        "valid": True,
        "validation_type": "mandatory_fields",
        "errors": [],
        "warnings": ["Date format could be clearer"],
        "confidence": 0.98
    })
    
    return schema


def create_workflow_result_schema() -> OutputSchema:
    """Schema for complete workflow result"""
    schema = OutputSchema(
        name="WorkflowResult",
        description="Result from workflow execution"
    )
    
    schema.add_field(SchemaField(
        "workflow_id", FieldType.STRING,
        "ID of executed workflow"
    ))
    
    schema.add_field(SchemaField(
        "status", FieldType.STRING,
        "Workflow status (completed, failed, etc.)",
        enum=["completed", "failed", "paused"]
    ))
    
    schema.add_field(SchemaField(
        "results", FieldType.OBJECT,
        "Step-by-step results from workflow"
    ))
    
    schema.add_field(SchemaField(
        "duration_ms", FieldType.INTEGER,
        "Total execution time in milliseconds"
    ))
    
    schema.add_field(SchemaField(
        "error", FieldType.STRING,
        "Error message if workflow failed", required=False
    ))
    
    schema.add_example({
        "workflow_id": "workflow_doc_analysis",
        "status": "completed",
        "results": {
            "step_extract": {"fields": {...}},
            "step_validate": {"valid": True},
            "step_summarize": {"summary": "..."}
        },
        "duration_ms": 2500,
        "error": None
    })
    
    return schema

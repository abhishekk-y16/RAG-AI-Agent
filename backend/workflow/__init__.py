"""
Workflow Module - Phase 5: Workflow Automation & Structured Outputs
Contains workflow definitions, execution engine, tools, and schemas
"""

from .engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStep,
    create_document_analysis_workflow,
    create_qa_workflow,
    create_extraction_validation_workflow,
    WorkflowStatus,
    StepStatus
)

from .tools import (
    ToolFactory,
    TextExtractionTool,
    TextValidationTool,
    TextSummarizationTool,
    DocumentClassificationTool,
    DataCleaningTool,
    ComparisonTool,
    ToolResult
)

from .schemas import (
    OutputFormatter,
    OutputSchema,
    SchemaField,
    FieldType,
    create_extraction_schema,
    create_qa_schema,
    create_summary_schema,
    create_validation_schema,
    create_workflow_result_schema
)

__all__ = [
    # Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowStep",
    "WorkflowStatus",
    "StepStatus",
    "create_document_analysis_workflow",
    "create_qa_workflow",
    "create_extraction_validation_workflow",
    
    # Tools
    "ToolFactory",
    "TextExtractionTool",
    "TextValidationTool",
    "TextSummarizationTool",
    "DocumentClassificationTool",
    "DataCleaningTool",
    "ComparisonTool",
    "ToolResult",
    
    # Schemas
    "OutputFormatter",
    "OutputSchema",
    "SchemaField",
    "FieldType",
    "create_extraction_schema",
    "create_qa_schema",
    "create_summary_schema",
    "create_validation_schema",
    "create_workflow_result_schema"
]

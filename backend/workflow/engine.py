"""
Workflow Engine for Phase 5 - Multi-step workflows with structured outputs
Supports workflow definitions, execution, state management, and error recovery
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import uuid
from abc import ABC, abstractmethod


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OutputSchema(str, Enum):
    """Structured output formats"""
    JSON = "json"
    TABLE = "table"
    MARKDOWN = "markdown"
    TEXT = "text"


@dataclass
class StepInput:
    """Input definition for a workflow step"""
    name: str
    type: str  # str, int, float, bool, list, dict, document
    required: bool = True
    description: str = ""
    default: Any = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StepOutput:
    """Output definition for a workflow step"""
    name: str
    type: str  # Same types as input
    description: str = ""
    schema: Optional[Dict] = None  # Optional JSON schema
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowStep:
    """Definition of a single workflow step"""
    id: str
    name: str
    tool: str  # Tool to execute (extract, qa, summarize, etc.)
    inputs: List[StepInput] = field(default_factory=list)
    outputs: List[StepOutput] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    timeout_seconds: int = 300
    skip_on_error: bool = False
    depends_on: List[str] = field(default_factory=list)  # Step IDs this depends on
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "tool": self.tool,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "config": self.config,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds,
            "skip_on_error": self.skip_on_error,
            "depends_on": self.depends_on
        }


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    id: str
    name: str
    description: str
    version: str = "1.0"
    steps: List[WorkflowStep] = field(default_factory=list)
    output_schema: OutputSchema = OutputSchema.JSON
    parallel_steps: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "output_schema": self.output_schema.value,
            "parallel_steps": self.parallel_steps,
            "created_at": self.created_at
        }
    
    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow"""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get step by ID"""
        return next((s for s in self.steps if s.id == step_id), None)
    
    def get_root_steps(self) -> List[WorkflowStep]:
        """Get steps with no dependencies"""
        return [s for s in self.steps if not s.depends_on]


@dataclass
class StepExecution:
    """Execution record for a single step"""
    step_id: str
    step_name: str
    status: StepStatus
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowExecution:
    """Complete workflow execution record"""
    id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    step_executions: List[StepExecution] = field(default_factory=list)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    duration_ms: int = 0
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "step_executions": [s.to_dict() for s in self.step_executions],
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "user_id": self.user_id
        }
    
    def add_step_execution(self, step_exec: StepExecution) -> None:
        """Add a step execution"""
        self.step_executions.append(step_exec)
    
    def get_step_execution(self, step_id: str) -> Optional[StepExecution]:
        """Get step execution by step ID"""
        return next((s for s in self.step_executions if s.step_id == step_id), None)


class WorkflowTool(ABC):
    """Abstract base class for workflow tools"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the tool with given inputs"""
        pass
    
    def get_input_schema(self) -> List[StepInput]:
        """Define expected inputs"""
        return []
    
    def get_output_schema(self) -> List[StepOutput]:
        """Define expected outputs"""
        return []


class WorkflowEngine:
    """Executes workflows and manages state"""
    
    def __init__(self):
        self.tools: Dict[str, WorkflowTool] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.definitions: Dict[str, WorkflowDefinition] = {}
    
    def register_tool(self, tool: WorkflowTool) -> None:
        """Register a workflow tool"""
        self.tools[tool.name] = tool
    
    def register_workflow(self, definition: WorkflowDefinition) -> None:
        """Register a workflow definition"""
        self.definitions[definition.id] = definition
    
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any], 
                        context: Dict[str, Any] = None) -> WorkflowExecution:
        """
        Execute a complete workflow
        
        Args:
            workflow_id: ID of the workflow to execute
            inputs: Input data for the workflow
            context: Additional context (user_id, doc_id, etc.)
        
        Returns:
            Completed WorkflowExecution
        """
        # Get workflow definition
        definition = self.definitions.get(workflow_id)
        if not definition:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # Create execution record
        execution = WorkflowExecution(
            id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            workflow_name=definition.name,
            status=WorkflowStatus.RUNNING,
            context=context or {}
        )
        
        # Store execution
        self.executions[execution.id] = execution
        
        try:
            # Execute steps
            step_outputs = self._execute_steps(definition, inputs, execution)
            
            # Set output data
            execution.output_data = step_outputs
            execution.status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
        
        # Set completion time
        execution.completed_at = datetime.now().isoformat()
        
        return execution
    
    def _execute_steps(self, definition: WorkflowDefinition, 
                      inputs: Dict[str, Any], execution: WorkflowExecution) -> Dict[str, Any]:
        """Execute all steps in workflow"""
        
        outputs = {}
        executed_steps = set()
        step_results = {}
        
        # Execute steps in order of dependencies
        while len(executed_steps) < len(definition.steps):
            # Find next executable step
            next_step = None
            for step in definition.steps:
                if step.id not in executed_steps:
                    # Check dependencies
                    if all(dep in executed_steps for dep in step.depends_on):
                        next_step = step
                        break
            
            if not next_step:
                # Circular dependency
                raise ValueError("Circular dependency detected in workflow")
            
            # Execute step
            step_exec = self._execute_step(
                next_step, inputs, step_results, execution
            )
            
            execution.add_step_execution(step_exec)
            executed_steps.add(next_step.id)
            
            if step_exec.status == StepStatus.COMPLETED:
                step_results[next_step.id] = step_exec.output_data
                outputs[next_step.id] = step_exec.output_data
            elif not next_step.skip_on_error:
                raise ValueError(f"Step failed: {next_step.name}: {step_exec.error}")
        
        return outputs
    
    def _execute_step(self, step: WorkflowStep, workflow_inputs: Dict[str, Any],
                     previous_outputs: Dict[str, Any], 
                     execution: WorkflowExecution) -> StepExecution:
        """Execute a single step"""
        
        # Prepare input data
        step_input = self._prepare_step_input(step, workflow_inputs, previous_outputs)
        
        # Create step execution record
        step_exec = StepExecution(
            step_id=step.id,
            step_name=step.name,
            status=StepStatus.RUNNING,
            input_data=step_input,
            started_at=datetime.now().isoformat()
        )
        
        # Get tool
        tool = self.tools.get(step.tool)
        if not tool:
            step_exec.status = StepStatus.FAILED
            step_exec.error = f"Tool not found: {step.tool}"
            return step_exec
        
        # Execute with retries
        last_error = None
        for attempt in range(step.retry_count + 1):
            try:
                # Execute tool
                result = tool.execute(step_input, step.config)
                
                step_exec.status = StepStatus.COMPLETED
                step_exec.output_data = result
                step_exec.retry_count = attempt
                break
                
            except Exception as e:
                last_error = str(e)
                if attempt < step.retry_count:
                    continue
                else:
                    step_exec.status = StepStatus.FAILED
                    step_exec.error = last_error
        
        # Set completion time
        step_exec.completed_at = datetime.now().isoformat()
        
        return step_exec
    
    def _prepare_step_input(self, step: WorkflowStep, 
                           workflow_inputs: Dict[str, Any],
                           previous_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input data for a step"""
        
        result = {}
        
        for input_def in step.inputs:
            # Try to get from previous step outputs
            found = False
            for step_id, outputs in previous_outputs.items():
                if input_def.name in outputs:
                    result[input_def.name] = outputs[input_def.name]
                    found = True
                    break
            
            # Try to get from workflow inputs
            if not found and input_def.name in workflow_inputs:
                result[input_def.name] = workflow_inputs[input_def.name]
                found = True
            
            # Use default if available
            if not found and input_def.default is not None:
                result[input_def.name] = input_def.default
                found = True
            
            # Check if required
            if not found and input_def.required:
                raise ValueError(f"Required input not found: {input_def.name}")
        
        return result
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution record"""
        return self.executions.get(execution_id)
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition"""
        return self.definitions.get(workflow_id)
    
    def list_workflows(self) -> List[Dict]:
        """List all workflow definitions"""
        return [w.to_dict() for w in self.definitions.values()]
    
    def list_executions(self, workflow_id: str = None) -> List[Dict]:
        """List executions, optionally filtered by workflow"""
        result = []
        for exec in self.executions.values():
            if workflow_id is None or exec.workflow_id == workflow_id:
                result.append(exec.to_dict())
        return result


# Predefined Workflow Templates

def create_document_analysis_workflow() -> WorkflowDefinition:
    """Template: Extract fields, validate, and generate summary"""
    
    workflow = WorkflowDefinition(
        id="workflow_doc_analysis",
        name="Document Analysis",
        description="Extract fields, validate completeness, and generate summary",
        output_schema=OutputSchema.JSON
    )
    
    # Step 1: Extract fields
    extract_step = WorkflowStep(
        id="step_extract",
        name="Extract Fields",
        tool="extract_fields",
        inputs=[
            StepInput("document_text", "str", True, "Document text to analyze"),
            StepInput("document_type", "str", True, "Type of document")
        ],
        outputs=[
            StepOutput("extracted_fields", "dict", "Extracted field data"),
            StepOutput("extraction_confidence", "float", "Confidence score")
        ],
        config={"strict_mode": True}
    )
    workflow.add_step(extract_step)
    
    # Step 2: Validate
    validate_step = WorkflowStep(
        id="step_validate",
        name="Validate Document",
        tool="validate_document",
        inputs=[
            StepInput("extracted_fields", "dict", True, "Fields to validate")
        ],
        outputs=[
            StepOutput("validation_status", "str", "VALID or INVALID"),
            StepOutput("missing_fields", "list", "Required fields that are missing")
        ],
        depends_on=["step_extract"],
        skip_on_error=True
    )
    workflow.add_step(validate_step)
    
    # Step 3: Summarize
    summary_step = WorkflowStep(
        id="step_summarize",
        name="Generate Summary",
        tool="summarize",
        inputs=[
            StepInput("document_text", "str", True, "Text to summarize"),
            StepInput("max_length", "int", False, "Max summary length", default=500)
        ],
        outputs=[
            StepOutput("summary", "str", "Document summary")
        ],
        depends_on=["step_extract"]
    )
    workflow.add_step(summary_step)
    
    return workflow


def create_qa_workflow() -> WorkflowDefinition:
    """Template: Search, retrieve context, and answer question"""
    
    workflow = WorkflowDefinition(
        id="workflow_qa",
        name="Question Answering",
        description="Search documents, retrieve context, generate answer",
        output_schema=OutputSchema.JSON
    )
    
    # Step 1: Semantic search
    search_step = WorkflowStep(
        id="step_search",
        name="Semantic Search",
        tool="semantic_search",
        inputs=[
            StepInput("query", "str", True, "Search query"),
            StepInput("top_k", "int", False, "Number of results", default=5),
            StepInput("doc_id", "str", False, "Optional document filter")
        ],
        outputs=[
            StepOutput("search_results", "list", "List of relevant chunks")
        ]
    )
    workflow.add_step(search_step)
    
    # Step 2: Generate answer
    answer_step = WorkflowStep(
        id="step_answer",
        name="Generate Answer",
        tool="generate_answer",
        inputs=[
            StepInput("query", "str", True, "Original query"),
            StepInput("context", "list", True, "Search results as context")
        ],
        outputs=[
            StepOutput("answer", "str", "Generated answer"),
            StepOutput("sources", "list", "Source references")
        ],
        depends_on=["step_search"]
    )
    workflow.add_step(answer_step)
    
    return workflow


def create_extraction_validation_workflow() -> WorkflowDefinition:
    """Template: Extract, validate, and generate report"""
    
    workflow = WorkflowDefinition(
        id="workflow_extract_validate",
        name="Extraction & Validation",
        description="Extract structured data and validate against schema",
        output_schema=OutputSchema.JSON,
        parallel_steps=False
    )
    
    # Step 1: Extract
    extract_step = WorkflowStep(
        id="step_extract",
        name="Extract Data",
        tool="extract_fields",
        inputs=[
            StepInput("content", "str", True, "Text content"),
            StepInput("schema", "dict", True, "Target schema")
        ],
        outputs=[
            StepOutput("extracted", "dict", "Extracted data"),
            StepOutput("confidence", "float", "Extraction confidence")
        ]
    )
    workflow.add_step(extract_step)
    
    # Step 2: Validate
    validate_step = WorkflowStep(
        id="step_validate",
        name="Validate",
        tool="validate_schema",
        inputs=[
            StepInput("data", "dict", True, "Data to validate"),
            StepInput("schema", "dict", True, "Validation schema")
        ],
        outputs=[
            StepOutput("valid", "bool", "Validation result"),
            StepOutput("errors", "list", "Validation errors if any")
        ],
        depends_on=["step_extract"]
    )
    workflow.add_step(validate_step)
    
    # Step 3: Report
    report_step = WorkflowStep(
        id="step_report",
        name="Generate Report",
        tool="generate_report",
        inputs=[
            StepInput("extracted", "dict", True, "Extracted data"),
            StepInput("validation_result", "dict", True, "Validation results")
        ],
        outputs=[
            StepOutput("report", "str", "Formatted report")
        ],
        depends_on=["step_extract", "step_validate"]
    )
    workflow.add_step(report_step)
    
    return workflow

import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

# Load environment variables from .env file
load_dotenv()

from rag.pdf_to_text import pdf_to_text
from rag.chunking import chunk_text
from rag.field_extractor import extract_fields_from_text, get_mandatory_fields_for_type
from rag.embed_store import build_and_save_index, load_index, search_index
from rag.metadata_manager import MetadataManager
from rag.rag_answer import retrieve, generate_answer
from rag.semantic_rag import create_rag_system
from agent.intent_classifier import classify_intent, get_tool_for_intent
from agent.orchestrator import ToolOrchestrator
from agent.tools import DocumentTools
from db.database import DatabaseManager
from workflow.engine import WorkflowEngine, WorkflowDefinition
from workflow.tools import ToolFactory
from workflow.schemas import (
    OutputFormatter, create_extraction_schema, create_qa_schema,
    create_summary_schema, create_validation_schema
)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq API
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_PATH = os.path.join(DATA_DIR, "knowledge.pdf")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "chunks.json")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "vector_store.json")

# Initialize global state
index = None
chunks = None
metadata_manager = MetadataManager(METADATA_PATH)
db_manager = DatabaseManager(os.path.join(DATA_DIR, "ragaiagent.db"))  # SQLite database

# Initialize semantic RAG system (Phase 4)
try:
    semantic_rag = create_rag_system(use_hybrid=True)
    print("✓ Semantic RAG system initialized")
except Exception as e:
    print(f"⚠ Semantic RAG initialization warning: {e}")
    semantic_rag = None

# Initialize workflow engine (Phase 5)
workflow_engine = WorkflowEngine()
tool_factory = ToolFactory()

# Store for workflow definitions
stored_workflows: Dict[str, WorkflowDefinition] = {}

class ChatIn(BaseModel):
    message: str
    document_id: Optional[str] = None  # Specific doc to search


class UploadResponse(BaseModel):
    status: str
    filename: str
    document_id: str
    pages: int
    chunks: int
    uploadedAt: str


@app.post("/ingest")
def ingest():
    """Ingest the default PDF at data/knowledge.pdf"""
    global index, chunks

    text, _ = pdf_to_text(PDF_PATH)
    chunks_list = chunk_text(text)
    
    # Convert chunks to format expected by build_and_save_index
    formatted_chunks = []
    for i, chunk in enumerate(chunks_list):
        chunk_dict = {
            "chunk_id": f"chunk_{i}",
            "doc_id": "default",
            "text": chunk["text"],
            "position": chunk["position"],
            "extracted_fields": chunk["extracted_fields"]
        }
        formatted_chunks.append(chunk_dict)
    
    build_and_save_index(formatted_chunks, INDEX_PATH, META_PATH)
    index, chunks = load_index(INDEX_PATH, META_PATH)

    return {"status": "ok", "chunks": len(formatted_chunks)}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF file (multi-document support)"""
    global index, chunks
    
    # Support PDF and DOCX
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx") or file.filename.endswith(".txt")):
        return {"error": "Only PDF, DOCX, and TXT files are allowed"}
    
    try:
        # Create temporary file for processing
        temp_path = os.path.join(DATA_DIR, "temp_" + file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text and metadata
        text, page_count = pdf_to_text(temp_path)
        chunks_list = chunk_text(text)
        
        # Create document record in metadata
        doc_id = metadata_manager.add_document(
            filename=file.filename,
            source_type="insurance_claim",  # Could be detected from filename/content
            page_count=page_count
        )
        
        # Add chunks to metadata and build formatted chunks
        formatted_chunks = []
        for i, chunk in enumerate(chunks_list):
            chunk_dict = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk["text"],
                "page": 1,  # Could be improved with page detection
                "position": chunk["position"],
                "token_count": chunk["token_count"],
                "extracted_fields": chunk["extracted_fields"]
            }
            formatted_chunks.append(chunk_dict)
            
            # Also add to metadata manager
            metadata_manager.add_chunk(
                doc_id=doc_id,
                chunk_text=chunk["text"],
                page=1,
                token_count=chunk["token_count"],
                extracted_fields=chunk["extracted_fields"]
            )
        
        # Validate document
        doc_fields = {}
        for chunk in formatted_chunks:
            doc_fields.update(chunk["extracted_fields"])
        
        mandatory_fields = get_mandatory_fields_for_type("insurance_claim")
        missing_fields = [f for f in mandatory_fields if f not in doc_fields]
        quality_score = 1.0 - (len(missing_fields) / len(mandatory_fields)) if mandatory_fields else 1.0
        
        metadata_manager.update_validation(
            doc_id=doc_id,
            mandatory_fields=mandatory_fields,
            missing_fields=missing_fields,
            quality_score=quality_score
        )
        
        # Update index (rebuild with all docs)
        all_chunks = metadata_manager.get_all_chunks()
        all_formatted_chunks = []
        for chunk_text, d_id, c_id in all_chunks:
            all_formatted_chunks.append({
                "chunk_id": c_id,
                "doc_id": d_id,
                "text": chunk_text
            })
        
        build_and_save_index(all_formatted_chunks, INDEX_PATH, META_PATH)
        index, chunks = load_index(INDEX_PATH, META_PATH)
        
        # Update processing status
        metadata_manager.update_processing_status(doc_id, "completed")
        
        # Save to SQLite database
        db_manager.save_document(
            doc_id=doc_id,
            filename=file.filename,
            source="insurance_claim",
            file_size=os.path.getsize(temp_path) if os.path.exists(temp_path) else 0,
            quality_score=quality_score
        )
        db_manager.update_document(doc_id, {"chunk_count": len(formatted_chunks)})
        
        # Log upload event
        event_id = str(uuid.uuid4())
        db_manager.log_event(
            event_id=event_id,
            doc_id=doc_id,
            event_type="upload",
            description=f"Document uploaded: {file.filename}",
            metadata=f'{{"filename": "{file.filename}", "page_count": {page_count}, "chunk_count": {len(formatted_chunks)}}}'
        )
        
        # Save validation result
        val_id = str(uuid.uuid4())
        db_manager.save_validation(
            val_id=val_id,
            doc_id=doc_id,
            validation_type="mandatory_fields",
            status="passed" if not missing_fields else "failed",
            details=f"Missing: {','.join(missing_fields)}",
            confidence=quality_score
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "status": "ok",
            "filename": file.filename,
            "document_id": doc_id,
            "pages": page_count,
            "chunks": len(formatted_chunks),
            "uploadedAt": datetime.now().isoformat(),
            "quality_score": quality_score,
            "missing_fields": missing_fields
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
def chat(payload: ChatIn):
    """Chat with the documents (intent-aware)"""
    global index, chunks

    if index is None or chunks is None:
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            index, chunks = load_index(INDEX_PATH, META_PATH)
        else:
            return {"answer": "Knowledge base not ingested yet. Call /ingest first.", "intent": "error"}

    # Classify query intent
    intent, confidence = classify_intent(payload.message)
    tool_name = get_tool_for_intent(intent)
    
    # Retrieve relevant chunks
    if chunks:
        hits = search_index(index, payload.message, chunks, k=4)
        hit_texts = [h[0]["text"] for h in hits]
    else:
        hit_texts = []
    
    # Execute the appropriate tool based on intent
    hit_chunks = hits if 'hits' in locals() else []
    all_chunks_for_tools = chunks if chunks else []
    
    tool_response = ToolOrchestrator.execute(
        intent=intent,
        query=payload.message,
        hit_chunks=hit_chunks,
        all_chunks=all_chunks_for_tools,
        mandatory_fields=get_mandatory_fields_for_type("insurance_claim")
    )
    
    # Prepare sources from hits
    sources = []
    if chunks and hit_chunks:
        for hit_chunk, confidence_score in hit_chunks:
            sources.append({
                "chunk_id": hit_chunk.get("chunk_id", "unknown"),
                "doc_id": hit_chunk.get("doc_id", "unknown"),
                "confidence": float(confidence_score)
            })
    
    # Combine tool output with sources
    structured_data = tool_response.structured_data or {}
    
    # Log query to database
    first_doc_id = sources[0].get("doc_id", "unknown") if sources else "unknown"
    
    query_id = str(uuid.uuid4())
    db_manager.log_query(
        query_id=query_id,
        doc_id=first_doc_id if first_doc_id != "unknown" else None,
        query_text=payload.message,
        intent=intent.value,
        model_used=tool_response.tool_name,
        response_text=tool_response.answer,
        confidence=float(tool_response.confidence),
        processing_time_ms=0
    )
    
    # Log as audit event
    event_id = str(uuid.uuid4())
    db_manager.log_event(
        event_id=event_id,
        doc_id=first_doc_id if first_doc_id != "unknown" else None,
        event_type="query",
        description=f"Query executed with intent: {intent.value}",
        metadata=f'{{"intent": "{intent.value}", "tool": "{tool_response.tool_name}"}}'
    )
    
    response = {
        "answer": tool_response.answer,
        "intent": intent.value,
        "intent_confidence": confidence,
        "tool": tool_response.tool_name,
        "sources": sources,
        "confidence": tool_response.confidence,
        "structured_data": structured_data,
        "metadata": tool_response.metadata
    }
    
    return response


@app.post("/search/semantic")
def semantic_search(query: str, doc_id: Optional[str] = None, top_k: int = 5):
    """
    Semantic search using real embeddings (Phase 4)
    Returns semantically similar chunks ranked by similarity score
    """
    if semantic_rag is None:
        return {"error": "Semantic RAG not initialized"}
    
    try:
        results = semantic_rag.retrieve(query, doc_id=doc_id, top_k=top_k)
        
        # Log search event
        event_id = str(uuid.uuid4())
        db_manager.log_event(
            event_id=event_id,
            doc_id=doc_id,
            event_type="semantic_search",
            description=f"Semantic search executed",
            metadata=f'{{"query_length": {len(query)}, "doc_id": "{doc_id}"}}'
        )
        
        return {
            "query": query,
            "doc_id": doc_id,
            "results": results,
            "count": len(results),
            "provider": semantic_rag.embedding_service.provider_name
        }
    except Exception as e:
        return {"error": f"Semantic search failed: {str(e)}"}


@app.post("/rag/answer")
def rag_answer(query: str, doc_id: Optional[str] = None, top_k: int = 5):
    """
    RAG-based answer generation using semantic search
    """
    if semantic_rag is None:
        return {"error": "Semantic RAG not initialized"}
    
    try:
        result = semantic_rag.answer_question(query, doc_id=doc_id, top_k=top_k)
        
        # Log RAG query
        query_id = str(uuid.uuid4())
        db_manager.log_query(
            query_id=query_id,
            doc_id=doc_id,
            query_text=query,
            intent="rag_query",
            model_used="semantic_rag",
            response_text=result["answer"],
            confidence=0.85
        )
        
        return result
    except Exception as e:
        return {"error": f"RAG answer generation failed: {str(e)}"}


@app.get("/search/stats")
def search_stats():
    """Get semantic search engine statistics"""
    if semantic_rag is None:
        return {"error": "Semantic RAG not initialized"}
    
    return semantic_rag.get_stats()


@app.get("/documents")
def get_documents():
    """Get all uploaded documents"""
    docs = metadata_manager.get_all_documents()
    return {
        "documents": [
            {
                "id": d["id"],
                "filename": d["filename"],
                "source_type": d["source_type"],
                "uploaded_at": d["uploaded_at"],
                "page_count": d["page_count"],
                "chunk_count": len(d["chunks"]),
                "validation": d["validation"]
            }
            for d in docs
        ]
    }


@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    """Get specific document details"""
    doc = metadata_manager.get_document(doc_id)
    if not doc:
        return {"error": "Document not found"}
    
    return {
        "id": doc["id"],
        "filename": doc["filename"],
        "source_type": doc["source_type"],
        "uploaded_at": doc["uploaded_at"],
        "page_count": doc["page_count"],
        "chunks": len(doc["chunks"]),
        "processing": doc["processing"],
        "validation": doc["validation"],
        "extracted_fields": {
            field: value 
            for chunk in doc["chunks"] 
            for field, value in chunk["extracted_fields"].items()
        }
    }


@app.post("/validate")
def validate_document():
    """
    Validate all documents for completeness and data quality.
    Returns validation report for each document.
    """
    global index, chunks
    
    if not chunks:
        return {"error": "No documents to validate"}
    
    # Validate using tool
    validation_result = DocumentTools.validate_fields(
        "Validate all documents",
        [],
        chunks,
        get_mandatory_fields_for_type("insurance_claim")
    )
    
    return {
        "status": "completed",
        "report": validation_result.answer,
        "structured_data": validation_result.structured_data,
        "confidence": validation_result.confidence
    }


@app.post("/risk-analysis")
def analyze_risks():
    """
    Analyze documents for potential risks and anomalies.
    Returns risk report with severity levels.
    """
    global index, chunks
    
    if not chunks:
        return {"error": "No documents to analyze"}
    
    # Analyze risks using tool
    risk_result = DocumentTools.risk_analysis(
        "Analyze all documents for risks",
        [],
        chunks
    )
    
    return {
        "status": "completed",
        "report": risk_result.answer,
        "structured_data": risk_result.structured_data,
        "confidence": risk_result.confidence
    }


@app.post("/workflow")
def generate_workflow():
    """
    Generate recommended workflow and next steps.
    Returns step-by-step processing workflow.
    """
    global index, chunks
    
    if not chunks:
        return {"error": "No documents to process"}
    
    # Generate workflow using tool
    workflow_result = DocumentTools.workflow_generate(
        "Generate workflow for documents",
        [],
        chunks
    )
    
    return {
        "status": "completed",
        "workflow": workflow_result.answer,
        "structured_data": workflow_result.structured_data,
        "confidence": workflow_result.confidence
    }


@app.post("/extract")
def extract_data(format: str = "json"):
    """
    Extract structured data from documents.
    Supports: json, csv, markdown
    """
    global index, chunks
    
    if not chunks:
        return {"error": "No documents to extract"}
    
    # Extract data using tool
    extraction_result = DocumentTools.extract_structured_data(
        "Extract all data",
        [],
        chunks
    )
    
    # Format output
    if format == "json":
        return extraction_result.structured_data
    elif format == "csv":
        # Convert to CSV format
        data = extraction_result.structured_data.get("extracted_fields", {})
        csv_lines = ["field,value"]
        for k, v in data.items():
            csv_lines.append(f'"{k}","{v}"')
        return {
            "format": "csv",
            "data": "\n".join(csv_lines)
        }
    else:  # markdown
        return {
            "format": "markdown",
            "data": extraction_result.answer
        }


@app.get("/analytics/document/{doc_id}")
def get_document_analytics(doc_id: str):
    """Get analytics for a specific document"""
    return db_manager.get_document_analytics(doc_id)


@app.get("/analytics/system")
def get_system_analytics():
    """Get system-wide analytics"""
    return db_manager.get_system_analytics()


@app.get("/audit-log/document/{doc_id}")
def get_document_audit_log(doc_id: str):
    """Get audit trail for a document"""
    events = db_manager.get_events_for_document(doc_id)
    return {
        "document_id": doc_id,
        "total_events": len(events),
        "events": events
    }


@app.get("/audit-log/all")
def get_all_audit_logs(limit: int = 50):
    """Get all audit logs"""
    events = db_manager.get_all_events()[:limit]
    return {
        "total_events": len(events),
        "events": events
    }


@app.get("/query-history/{doc_id}")
def get_query_history(doc_id: str, limit: int = 10):
    """Get query history for a document"""
    queries = db_manager.get_queries_for_document(doc_id, limit=limit)
    return {
        "document_id": doc_id,
        "total_queries": len(queries),
        "queries": queries
    }


@app.get("/risks/document/{doc_id}")
def get_document_risks(doc_id: str):
    """Get risk flags for a document"""
    risks = db_manager.get_risks_for_document(doc_id)
    return {
        "document_id": doc_id,
        "total_risks": len(risks),
        "high_severity": len([r for r in risks if r['severity'] == 'high']),
        "risks": risks
    }


@app.get("/risks/high-priority")
def get_high_priority_risks():
    """Get all high-severity risk flags across all documents"""
    risks = db_manager.get_high_risk_flags()
    return {
        "total_high_risk": len(risks),
        "risks": risks
    }


@app.get("/validations/document/{doc_id}")
def get_document_validations(doc_id: str):
    """Get validation history for a document"""
    validations = db_manager.get_validations_for_document(doc_id)
    return {
        "document_id": doc_id,
        "total_validations": len(validations),
        "latest": validations[0] if validations else None,
        "history": validations
    }


@app.get("/")
def read_root():
    return {"message": "RAG AI Agent Backend is running", "version": "4.0 (Level-5 with Phase 5: Workflow Automation & Structured Outputs)"}


# ==================== PHASE 5: WORKFLOW ENDPOINTS ====================

@app.get("/workflows/available")
def list_available_workflows():
    """List all available workflow templates"""
    return {
        "count": len(stored_workflows),
        "workflows": {
            name: {
                "id": wf.id,
                "description": wf.description,
                "steps": len(wf.steps),
                "inputs": list(wf.inputs.keys()) if wf.inputs else []
            }
            for name, wf in stored_workflows.items()
        },
        "templates": {
            "document_analysis": "Extract, validate, and summarize documents",
            "qa_workflow": "Answer questions using document context",
            "extraction_validation": "Extract fields and validate data quality"
        }
    }


@app.post("/workflows/register")
def register_workflow(workflow_def: Dict[str, Any]):
    """Register a new workflow definition"""
    try:
        # Create workflow from definition
        workflow = WorkflowDefinition(
            name=workflow_def.get("name"),
            description=workflow_def.get("description", ""),
            inputs=workflow_def.get("inputs", {}),
            outputs=workflow_def.get("outputs", {}),
            steps=[]  # Steps would be created separately
        )
        
        stored_workflows[workflow.name] = workflow
        
        db_manager.log_event(
            "workflow_registered",
            {"workflow_id": workflow.id, "workflow_name": workflow.name}
        )
        
        return {
            "status": "registered",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/{workflow_name}/execute")
def execute_workflow(workflow_name: str, inputs: Dict[str, Any]):
    """Execute a registered workflow"""
    try:
        if workflow_name not in stored_workflows:
            return {"status": "error", "message": f"Workflow '{workflow_name}' not found"}
        
        workflow = stored_workflows[workflow_name]
        
        # Execute workflow
        execution = workflow_engine.execute_workflow(workflow, inputs)
        
        # Log execution
        db_manager.log_event(
            "workflow_executed",
            {
                "workflow_id": workflow.id,
                "execution_id": execution.id,
                "status": execution.status.value
            }
        )
        
        return {
            "execution_id": execution.id,
            "workflow_id": workflow.id,
            "status": execution.status.value,
            "results": execution.step_results,
            "duration_ms": execution.duration_ms,
            "error": execution.error
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/templates/document-analysis")
def create_document_analysis_workflow():
    """Create and register document analysis workflow"""
    try:
        from workflow.engine import create_document_analysis_workflow
        
        workflow = create_document_analysis_workflow()
        stored_workflows["document_analysis"] = workflow
        
        return {
            "status": "created",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "steps": len(workflow.steps)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/templates/qa")
def create_qa_workflow():
    """Create and register Q&A workflow"""
    try:
        from workflow.engine import create_qa_workflow
        
        workflow = create_qa_workflow()
        stored_workflows["qa"] = workflow
        
        return {
            "status": "created",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "steps": len(workflow.steps)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/templates/extraction-validation")
def create_extraction_validation_workflow():
    """Create and register extraction & validation workflow"""
    try:
        from workflow.engine import create_extraction_validation_workflow
        
        workflow = create_extraction_validation_workflow()
        stored_workflows["extraction_validation"] = workflow
        
        return {
            "status": "created",
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "steps": len(workflow.steps)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/quick-extract")
def quick_extract_workflow(text: str, fields: list[str]):
    """Quick workflow to extract fields from text"""
    try:
        from workflow.tools import TextExtractionTool
        
        tool = TextExtractionTool()
        result = tool.execute(text, fields)
        
        # Format output
        formatter = OutputFormatter()
        
        return {
            "success": result.success,
            "extraction_result": result.data,
            "formatted": {
                "json": result.to_dict(),
                "markdown": formatter.to_markdown(
                    result.data or {},
                    "Extraction Results"
                )
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/quick-validate")
def quick_validate_workflow(data: Dict[str, Any], rules: Dict[str, Any]):
    """Quick workflow to validate data"""
    try:
        from workflow.tools import TextValidationTool
        
        tool = TextValidationTool()
        result = tool.execute(data, rules)
        
        formatter = OutputFormatter()
        
        return {
            "success": result.success,
            "validation_result": result.data,
            "formatted": {
                "json": result.to_dict(),
                "table": formatter.to_table(result.data or {})
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/quick-summarize")
def quick_summarize_workflow(text: str, max_sentences: int = 3):
    """Quick workflow to summarize text"""
    try:
        from workflow.tools import TextSummarizationTool
        
        tool = TextSummarizationTool()
        result = tool.execute(text, max_sentences)
        
        formatter = OutputFormatter()
        
        return {
            "success": result.success,
            "summary_result": result.data,
            "formatted": {
                "text": formatter.to_text(result.data or {}, "Summary"),
                "markdown": formatter.to_markdown(result.data or {})
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/workflows/quick-classify")
def quick_classify_workflow(text: str, categories: Optional[list[str]] = None):
    """Quick workflow to classify documents"""
    try:
        from workflow.tools import DocumentClassificationTool
        
        tool = DocumentClassificationTool()
        result = tool.execute(text, categories)
        
        return {
            "success": result.success,
            "classification_result": result.data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/tools/available")
def get_available_tools():
    """Get information about available workflow tools"""
    return {
        "available_tools": tool_factory.list_tools(),
        "tool_info": tool_factory.get_tool_info()
    }


@app.post("/schemas/validate")
def validate_against_schema(schema_name: str, data: Dict[str, Any]):
    """Validate data against a predefined schema"""
    try:
        schemas = {
            "extraction": create_extraction_schema(),
            "qa": create_qa_schema(),
            "summary": create_summary_schema(),
            "validation": create_validation_schema()
        }
        
        if schema_name not in schemas:
            return {"valid": False, "error": f"Schema '{schema_name}' not found"}
        
        schema = schemas[schema_name]
        is_valid, error = schema.validate(data)
        
        return {
            "schema": schema_name,
            "valid": is_valid,
            "error": error,
            "schema_info": {
                "name": schema.name,
                "description": schema.description,
                "fields": len(schema.fields)
            }
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.get("/schemas/available")
def get_available_schemas():
    """List available output schemas"""
    schemas = {
        "extraction": create_extraction_schema(),
        "qa": create_qa_schema(),
        "summary": create_summary_schema(),
        "validation": create_validation_schema()
    }
    
    return {
        "available_schemas": list(schemas.keys()),
        "schemas": {
            name: {
                "description": schema.description,
                "fields": len(schema.fields),
                "examples": len(schema.examples)
            }
            for name, schema in schemas.items()
        }
    }

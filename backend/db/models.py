"""
Database models for RAG agent persistence.
Uses SQLAlchemy + SQLite for development (easily upgradeable to PostgreSQL).
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

# For development, we'll use file-based models
# In production, this would use SQLAlchemy with database backend

class Document:
    """Represents an uploaded document"""
    def __init__(self, doc_id: str, filename: str, source_type: str, 
                 page_count: int, uploaded_at: datetime):
        self.id = doc_id
        self.filename = filename
        self.source_type = source_type
        self.page_count = page_count
        self.uploaded_at = uploaded_at
        self.created_at = datetime.now()
        self.status = "completed"  # processing, completed, error
        self.error_message = None
        self.quality_score = 0.0
        self.chunk_count = 0
    
    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "source_type": self.source_type,
            "page_count": self.page_count,
            "uploaded_at": self.uploaded_at.isoformat(),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "error_message": self.error_message,
            "quality_score": self.quality_score,
            "chunk_count": self.chunk_count
        }


class ValidationResult:
    """Validation result for a document"""
    def __init__(self, doc_id: str, mandatory_fields: List[str], 
                 missing_fields: List[str], quality_score: float):
        self.id = f"val_{doc_id}_{datetime.now().timestamp()}"
        self.doc_id = doc_id
        self.timestamp = datetime.now()
        self.mandatory_fields = mandatory_fields
        self.missing_fields = missing_fields
        self.quality_score = quality_score
        self.status = "incomplete" if missing_fields else "complete"
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "mandatory_fields": self.mandatory_fields,
            "missing_fields": self.missing_fields,
            "quality_score": self.quality_score,
            "status": self.status
        }


class RiskFlag:
    """Risk flag detected during analysis"""
    def __init__(self, doc_id: str, flag: str, severity: str, 
                 description: str, recommendation: str):
        self.id = f"risk_{doc_id}_{datetime.now().timestamp()}"
        self.doc_id = doc_id
        self.timestamp = datetime.now()
        self.flag = flag
        self.severity = severity  # high, medium, low
        self.description = description
        self.recommendation = recommendation
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "flag": self.flag,
            "severity": self.severity,
            "description": self.description,
            "recommendation": self.recommendation
        }


class ProcessingEvent:
    """Audit log entry for document processing"""
    def __init__(self, doc_id: str, event_type: str, details: Dict[str, Any],
                 status: str = "success", error: str = None):
        self.id = f"event_{doc_id}_{datetime.now().timestamp()}"
        self.doc_id = doc_id
        self.timestamp = datetime.now()
        self.event_type = event_type  # upload, validation, risk_analysis, query, etc
        self.details = details
        self.status = status  # success, pending, error
        self.error = error
        self.user = "anonymous"  # For future multi-user support
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "details": self.details,
            "status": self.status,
            "error": self.error,
            "user": self.user
        }


class QueryLog:
    """Log of user queries and AI responses"""
    def __init__(self, doc_id: str, query: str, intent: str, 
                 answer: str, tool: str, confidence: float, 
                 sources: List[Dict] = None):
        self.id = f"query_{doc_id}_{datetime.now().timestamp()}"
        self.doc_id = doc_id
        self.timestamp = datetime.now()
        self.query = query
        self.intent = intent
        self.answer = answer
        self.tool = tool
        self.confidence = confidence
        self.sources = sources or []
        self.user_feedback = None  # helpful, not_helpful, or None
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "intent": self.intent,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "tool": self.tool,
            "confidence": self.confidence,
            "sources": self.sources,
            "user_feedback": self.user_feedback
        }


class ExtractionResult:
    """Extracted data from document"""
    def __init__(self, doc_id: str, extracted_data: Dict[str, Any]):
        self.id = f"extract_{doc_id}_{datetime.now().timestamp()}"
        self.doc_id = doc_id
        self.timestamp = datetime.now()
        self.extracted_data = extracted_data
        self.field_count = len(extracted_data)
        self.completeness_score = 0.0
    
    def to_dict(self):
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp.isoformat(),
            "extracted_data": self.extracted_data,
            "field_count": self.field_count,
            "completeness_score": self.completeness_score
        }

"""
Manage document metadata for multi-document support.
Tracks: document IDs, chunk associations, extraction results, processing history.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid


class MetadataManager:
    """
    Manages document metadata for multi-document RAG system.
    
    Structure:
    {
        "documents": [
            {
                "id": "doc_uuid",
                "filename": "claim_form.pdf",
                "source_type": "insurance_claim",
                "uploaded_at": "2026-01-31T10:30:00",
                "file_hash": "sha256...",
                "page_count": 5,
                "chunks": [
                    {
                        "chunk_id": "chunk_uuid",
                        "text": "...",
                        "page": 1,
                        "start_pos": 0,
                        "end_pos": 1500,
                        "token_count": 450,
                        "extracted_fields": {
                            "policy_number": "...",
                            "claim_amount": 50000,
                            ...
                        }
                    }
                ],
                "processing": {
                    "status": "completed",
                    "started_at": "2026-01-31T10:30:00",
                    "completed_at": "2026-01-31T10:30:30",
                    "errors": []
                },
                "validation": {
                    "mandatory_fields": ["policy_number", "claim_amount"],
                    "missing_fields": ["ifsc_code"],
                    "quality_score": 0.85
                }
            }
        ]
    }
    """
    
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.data = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk, create if doesn't exist"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"documents": []}
        return {"documents": []}
    
    def _save_metadata(self):
        """Save metadata to disk"""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_document(self, filename: str, source_type: str = "document", 
                     file_hash: str = "", page_count: int = 0) -> str:
        """
        Add a new document to metadata.
        Returns: document_id
        """
        doc_id = str(uuid.uuid4())
        
        document = {
            "id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "uploaded_at": datetime.now().isoformat(),
            "file_hash": file_hash,
            "page_count": page_count,
            "chunks": [],
            "processing": {
                "status": "processing",
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "errors": []
            },
            "validation": {
                "mandatory_fields": [],
                "missing_fields": [],
                "quality_score": 0.0
            }
        }
        
        self.data["documents"].append(document)
        self._save_metadata()
        return doc_id
    
    def add_chunk(self, doc_id: str, chunk_text: str, page: int = 1,
                  start_pos: int = 0, end_pos: int = 0, 
                  token_count: int = 0, extracted_fields: Dict = None):
        """Add a chunk to a document's metadata"""
        doc = self._get_document(doc_id)
        if not doc:
            return
        
        chunk_id = str(uuid.uuid4())
        chunk = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "page": page,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "token_count": token_count,
            "extracted_fields": extracted_fields or {}
        }
        
        doc["chunks"].append(chunk)
        self._save_metadata()
        return chunk_id
    
    def update_processing_status(self, doc_id: str, status: str, 
                                 error: str = None):
        """Update processing status for a document"""
        doc = self._get_document(doc_id)
        if not doc:
            return
        
        doc["processing"]["status"] = status
        if status == "completed":
            doc["processing"]["completed_at"] = datetime.now().isoformat()
        
        if error:
            doc["processing"]["errors"].append({
                "timestamp": datetime.now().isoformat(),
                "error": error
            })
        
        self._save_metadata()
    
    def update_validation(self, doc_id: str, mandatory_fields: List[str],
                         missing_fields: List[str], quality_score: float):
        """Update validation results for a document"""
        doc = self._get_document(doc_id)
        if not doc:
            return
        
        doc["validation"] = {
            "mandatory_fields": mandatory_fields,
            "missing_fields": missing_fields,
            "quality_score": quality_score
        }
        self._save_metadata()
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID"""
        return self._get_document(doc_id)
    
    def _get_document(self, doc_id: str) -> Optional[Dict]:
        """Internal: Get document by ID"""
        for doc in self.data["documents"]:
            if doc["id"] == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        return self.data["documents"]
    
    def get_documents_by_type(self, source_type: str) -> List[Dict]:
        """Get all documents of a specific type"""
        return [d for d in self.data["documents"] if d["source_type"] == source_type]
    
    def get_all_chunks(self) -> List[tuple]:
        """
        Get all chunks from all documents.
        Returns: List of (chunk_text, doc_id, chunk_id)
        """
        chunks = []
        for doc in self.data["documents"]:
            for chunk in doc.get("chunks", []):
                chunks.append((chunk["text"], doc["id"], chunk["chunk_id"]))
        return chunks
    
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a specific document"""
        doc = self._get_document(doc_id)
        if not doc:
            return []
        return doc.get("chunks", [])
    
    def get_chunk_by_id(self, doc_id: str, chunk_id: str) -> Optional[Dict]:
        """Get a specific chunk"""
        doc = self._get_document(doc_id)
        if not doc:
            return None
        for chunk in doc.get("chunks", []):
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None

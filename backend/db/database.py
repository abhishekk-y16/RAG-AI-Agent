"""
SQLite Database manager for persistence and audit logging.
Provides scalable SQL-based storage for all RAG AI Agent operations.
"""
import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid




class DatabaseManager:
    """SQLite-based persistence manager for RAG AI Agent"""
    
    def __init__(self, db_path: str = "backend/data/ragaiagent.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Enable foreign keys
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        self._create_tables()
    
    def _create_tables(self):
        """Create all required SQLite tables"""
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                source TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size INTEGER,
                chunk_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.0,
                source_type TEXT DEFAULT 'pdf',
                status TEXT DEFAULT 'active'
            )
        """)

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                embedding BLOB,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Audit Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                event_type TEXT NOT NULL,
                description TEXT,
                user_agent TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Query History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                query_text TEXT NOT NULL,
                intent TEXT,
                model_used TEXT,
                response_text TEXT,
                confidence REAL,
                processing_time_ms INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Risk Flags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_flags (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                risk_type TEXT NOT NULL,
                severity TEXT,
                description TEXT,
                flagged_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved INTEGER DEFAULT 0,
                resolved_date TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Validations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validations (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                validation_type TEXT NOT NULL,
                status TEXT,
                details TEXT,
                confidence REAL,
                validated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Analytics Summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics_summary (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                total_queries INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                average_confidence REAL DEFAULT 0.0,
                last_queried TIMESTAMP,
                high_risk_count INTEGER DEFAULT 0,
                validation_status TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_doc ON audit_events(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON audit_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_doc ON query_history(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risks_doc ON risk_flags(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_risks_severity ON risk_flags(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_validations_doc ON validations(document_id)")
        
        self.conn.commit()
    
    # DOCUMENT OPERATIONS
    
    def save_document(self, doc_id: str, filename: str, source: str = None, 
                      file_size: int = 0, quality_score: float = 0.0) -> bool:
        """Save document record"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (id, filename, source, file_size, quality_score)
                VALUES (?, ?, ?, ?, ?)
            """, (doc_id, filename, source, file_size, quality_score))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving document: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Retrieve document record"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents ORDER BY upload_date DESC")
        return [dict(row) for row in cursor.fetchall()]
    
    def update_document(self, doc_id: str, updates: Dict) -> bool:
        """Update document record"""
        try:
            cursor = self.conn.cursor()
            allowed_fields = {'filename', 'source', 'file_size', 'chunk_count', 'quality_score', 'status'}
            
            for field, value in updates.items():
                if field in allowed_fields:
                    cursor.execute(f"UPDATE documents SET {field} = ? WHERE id = ?", (value, doc_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error updating document: {e}")
            return False
    
    # VALIDATION OPERATIONS
    
    def save_validation(self, val_id: str, doc_id: str, validation_type: str, 
                       status: str, details: str, confidence: float) -> bool:
        """Save validation result"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO validations 
                (id, document_id, validation_type, status, details, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (val_id, doc_id, validation_type, status, details, confidence))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving validation: {e}")
            return False
    
    def get_validations_for_document(self, doc_id: str) -> List[Dict]:
        """Get all validations for a document"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM validations 
            WHERE document_id = ? 
            ORDER BY validated_date DESC
        """, (doc_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_validation(self, doc_id: str) -> Optional[Dict]:
        """Get latest validation for document"""
        validations = self.get_validations_for_document(doc_id)
        return validations[0] if validations else None
    
    # RISK OPERATIONS
    
    def save_risk_flag(self, risk_id: str, doc_id: str, risk_type: str, 
                      severity: str, description: str) -> bool:
        """Save risk flag"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO risk_flags 
                (id, document_id, risk_type, severity, description)
                VALUES (?, ?, ?, ?, ?)
            """, (risk_id, doc_id, risk_type, severity, description))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving risk flag: {e}")
            return False
    
    def get_risks_for_document(self, doc_id: str) -> List[Dict]:
        """Get all risk flags for a document"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM risk_flags 
            WHERE document_id = ? 
            ORDER BY flagged_date DESC
        """, (doc_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_high_risk_flags(self) -> List[Dict]:
        """Get all high-severity risk flags"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM risk_flags 
            WHERE severity = 'high' AND resolved = 0
            ORDER BY flagged_date DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    # EVENT OPERATIONS (AUDIT LOG)
    
    def log_event(self, event_id: str, doc_id: str, event_type: str, 
                 description: str, metadata: str = None) -> bool:
        """Log processing event for audit trail"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO audit_events 
                (id, document_id, event_type, description, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (event_id, doc_id, event_type, description, metadata))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error logging event: {e}")
            return False
    
    def get_events_for_document(self, doc_id: str) -> List[Dict]:
        """Get audit trail for a document"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_events 
            WHERE document_id = ? 
            ORDER BY timestamp ASC
        """, (doc_id,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_all_events(self) -> List[Dict]:
        """Get all audit events"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_events 
            ORDER BY timestamp DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    # QUERY OPERATIONS
    
    def log_query(self, query_id: str, doc_id: str, query_text: str, 
                 intent: str = None, model_used: str = None, response_text: str = None,
                 confidence: float = 0.0, processing_time_ms: int = 0) -> bool:
        """Log user query and AI response"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO query_history 
                (id, document_id, query_text, intent, model_used, response_text, confidence, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_id, doc_id, query_text, intent, model_used, response_text, confidence, processing_time_ms))
            self.conn.commit()
            
            # Update analytics summary
            self._update_analytics_for_document(doc_id)
            return True
        except Exception as e:
            print(f"Error logging query: {e}")
            return False
    
    def get_queries_for_document(self, doc_id: str, limit: int = 10) -> List[Dict]:
        """Get query history for a document"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM query_history 
            WHERE document_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (doc_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    # ANALYTICS
    
    def _update_analytics_for_document(self, doc_id: str):
        """Update analytics summary for a document"""
        try:
            cursor = self.conn.cursor()
            
            # Get counts
            cursor.execute("SELECT COUNT(*) as count FROM query_history WHERE document_id = ?", (doc_id,))
            total_queries = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM chunks WHERE document_id = ?", (doc_id,))
            total_chunks = cursor.fetchone()['count']
            
            cursor.execute("SELECT AVG(confidence) as avg FROM query_history WHERE document_id = ?", (doc_id,))
            avg_confidence = cursor.fetchone()['avg'] or 0.0
            
            cursor.execute("SELECT COUNT(*) as count FROM risk_flags WHERE document_id = ? AND severity = 'high' AND resolved = 0", (doc_id,))
            high_risk_count = cursor.fetchone()['count']
            
            # Get latest validation status
            latest_val = self.get_latest_validation(doc_id)
            validation_status = latest_val['status'] if latest_val else 'unknown'
            
            # Get last query timestamp
            cursor.execute("SELECT timestamp FROM query_history WHERE document_id = ? ORDER BY timestamp DESC LIMIT 1", (doc_id,))
            last_query = cursor.fetchone()
            last_queried = last_query['timestamp'] if last_query else None
            
            # Insert or update analytics
            analytics_id = f"analytics_{doc_id}"
            cursor.execute("""
                INSERT OR REPLACE INTO analytics_summary 
                (id, document_id, total_queries, total_chunks, average_confidence, 
                 last_queried, high_risk_count, validation_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (analytics_id, doc_id, total_queries, total_chunks, avg_confidence,
                  last_queried, high_risk_count, validation_status))
            
            self.conn.commit()
        except Exception as e:
            print(f"Error updating analytics: {e}")
    
    def get_document_analytics(self, doc_id: str) -> Dict[str, Any]:
        """Get analytics for a specific document"""
        doc = self.get_document(doc_id)
        validations = self.get_validations_for_document(doc_id)
        risks = self.get_risks_for_document(doc_id)
        queries = self.get_queries_for_document(doc_id, limit=100)
        events = self.get_events_for_document(doc_id)
        
        if not doc:
            return {}
        
        return {
            "document": doc,
            "validation": {
                "latest": dict(validations[0]) if validations else None,
                "history_count": len(validations)
            },
            "risks": {
                "total": len(risks),
                "high_severity": len([r for r in risks if r['severity'] == 'high']),
                "medium_severity": len([r for r in risks if r['severity'] == 'medium']),
                "latest_flags": [dict(r) for r in risks[:5]]
            },
            "queries": {
                "total": len(queries),
                "by_intent": self._group_by_intent(queries),
                "average_confidence": self._avg_confidence(queries),
                "recent_queries": [dict(q) for q in queries[:5]]
            },
            "events": {
                "total": len(events),
                "by_type": self._group_by_event_type(events)
            }
        }
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get system-wide analytics"""
        all_docs = self.get_all_documents()
        all_events = self.get_all_events()
        all_risks = []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM risk_flags")
        all_risks = [dict(row) for row in cursor.fetchall()]
        
        return {
            "documents": {
                "total": len(all_docs),
                "by_source": self._group_by_source_type(all_docs),
                "average_quality": self._avg_quality_score(all_docs)
            },
            "risks": {
                "total": len(all_risks),
                "high_severity": len([r for r in all_risks if r['severity'] == 'high']),
                "documents_with_risks": len(set(r['document_id'] for r in all_risks if r['document_id']))
            },
            "events": {
                "total": len(all_events),
                "by_type": self._group_by_event_type(all_events)
            }
        }
    
    # HELPER METHODS
    
    @staticmethod
    def _group_by_intent(queries: List[Dict]) -> Dict[str, int]:
        """Group queries by intent"""
        groups = {}
        for q in queries:
            intent = q.get('intent', 'unknown')
            groups[intent] = groups.get(intent, 0) + 1
        return groups
    
    @staticmethod
    def _group_by_event_type(events: List[Dict]) -> Dict[str, int]:
        """Group events by type"""
        groups = {}
        for e in events:
            etype = e.get('event_type', 'unknown')
            groups[etype] = groups.get(etype, 0) + 1
        return groups
    
    @staticmethod
    def _group_by_source_type(docs: List[Dict]) -> Dict[str, int]:
        """Group documents by source type"""
        groups = {}
        for d in docs:
            stype = d.get('source_type', 'unknown')
            groups[stype] = groups.get(stype, 0) + 1
        return groups
    
    @staticmethod
    def _avg_confidence(queries: List[Dict]) -> float:
        """Calculate average confidence of queries"""
        if not queries:
            return 0.0
        confidences = [q.get('confidence', 0) for q in queries]
        return sum(confidences) / len(confidences)
    
    @staticmethod
    def _avg_quality_score(docs: List[Dict]) -> float:
        """Calculate average quality score"""
        if not docs:
            return 0.0
        scores = [d.get('quality_score', 0.0) for d in docs]
        return sum(scores) / len(scores)

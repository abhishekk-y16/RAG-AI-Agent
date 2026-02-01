"""
Tool implementations for the RAG agent.
Each tool corresponds to an intent type and performs specific document analysis tasks.
"""
import json
import re
from typing import Dict, List, Any, Tuple
from enum import Enum


class ToolResponse:
    """Structured response from a tool execution"""
    def __init__(self, answer: str, tool_name: str, structured_data: Any = None, 
                 confidence: float = 0.85, metadata: Dict = None):
        self.answer = answer
        self.tool_name = tool_name
        self.structured_data = structured_data
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "answer": self.answer,
            "tool": self.tool_name,
            "structured_data": self.structured_data,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class DocumentTools:
    """Collection of tools for document analysis"""
    
    @staticmethod
    def retrieve_specific_field(query: str, hit_chunks: List[Dict], 
                               all_chunks: List[Dict]) -> ToolResponse:
        """
        Extract specific facts/fields from documents.
        Tool for FACT intent.
        """
        if not hit_chunks:
            answer = "I couldn't find information relevant to your question in the documents."
            return ToolResponse(answer, "retrieve_specific_field", confidence=0.3)
        
        # Get most relevant chunk
        best_chunk = hit_chunks[0] if hit_chunks else None
        
        if best_chunk:
            # Extract fields from best match
            extracted_fields = best_chunk.get("extracted_fields", {})
            chunk_text = best_chunk.get("text", "")
            
            # Build answer with context
            answer = chunk_text[:500] + ("..." if len(chunk_text) > 500 else "")
            
            return ToolResponse(
                answer=answer,
                tool_name="retrieve_specific_field",
                structured_data={
                    "query": query,
                    "extracted_fields": extracted_fields,
                    "chunk_text": chunk_text
                },
                confidence=0.85
            )
        
        return ToolResponse("No relevant information found", "retrieve_specific_field", confidence=0.3)
    
    @staticmethod
    def generate_summary(query: str, hit_chunks: List[Dict], 
                        all_chunks: List[Dict]) -> ToolResponse:
        """
        Generate a summary of document(s).
        Tool for SUMMARY intent.
        """
        if not hit_chunks and not all_chunks:
            return ToolResponse("No documents to summarize", "generate_summary", confidence=0.2)
        
        # Use all chunks if available for full document summary
        chunks_to_summarize = all_chunks if all_chunks else hit_chunks
        
        # Collect key information
        summary_points = []
        extracted_fields_all = {}
        
        for chunk in chunks_to_summarize[:5]:  # Limit to first 5 chunks
            text = chunk.get("text", "")
            fields = chunk.get("extracted_fields", {})
            
            # Extract sentences
            sentences = re.split(r'[.!?]+', text)
            if sentences:
                summary_points.append(sentences[0].strip())
            
            extracted_fields_all.update(fields)
        
        # Build summary
        summary = "\n".join([f"• {s}" for s in summary_points if s])
        
        if not summary:
            summary = "This document contains insurance claim information."
        
        return ToolResponse(
            answer=summary,
            tool_name="generate_summary",
            structured_data={
                "summary_points": summary_points,
                "key_fields": extracted_fields_all,
                "chunk_count": len(chunks_to_summarize)
            },
            confidence=0.80
        )
    
    @staticmethod
    def extract_structured_data(query: str, hit_chunks: List[Dict], 
                               all_chunks: List[Dict]) -> ToolResponse:
        """
        Extract data in structured formats (JSON, table, etc).
        Tool for EXTRACTION intent.
        """
        chunks = all_chunks if all_chunks else hit_chunks
        
        if not chunks:
            return ToolResponse("No data to extract", "extract_structured_data", confidence=0.2)
        
        # Aggregate all extracted fields from all chunks
        aggregated_data = {}
        for chunk in chunks:
            aggregated_data.update(chunk.get("extracted_fields", {}))
        
        # Format as structured JSON
        structured_json = {
            "extraction_summary": {
                "document_count": 1,
                "chunks_processed": len(chunks),
                "fields_extracted": len(aggregated_data)
            },
            "extracted_fields": aggregated_data,
            "raw_data": aggregated_data
        }
        
        answer = f"""Extracted Data (JSON format):

```json
{json.dumps(structured_json, indent=2)}
```

**Fields Found:**
{chr(10).join(f"• {k}: {v}" for k, v in aggregated_data.items())}
"""
        
        return ToolResponse(
            answer=answer,
            tool_name="extract_structured_data",
            structured_data=structured_json,
            confidence=0.90
        )
    
    @staticmethod
    def validate_fields(query: str, hit_chunks: List[Dict], 
                       all_chunks: List[Dict], 
                       mandatory_fields: List[str] = None) -> ToolResponse:
        """
        Validate document completeness and data quality.
        Tool for VALIDATION intent.
        """
        if mandatory_fields is None:
            mandatory_fields = [
                "policy_number", "claim_amount", "incident_date",
                "policy_holder", "bank_account", "ifsc_code"
            ]
        
        chunks = all_chunks if all_chunks else hit_chunks
        
        if not chunks:
            return ToolResponse("No documents to validate", "validate_fields", confidence=0.2)
        
        # Aggregate all fields
        found_fields = {}
        for chunk in chunks:
            found_fields.update(chunk.get("extracted_fields", {}))
        
        # Check mandatory fields
        missing_fields = [f for f in mandatory_fields if f not in found_fields]
        present_fields = [f for f in mandatory_fields if f in found_fields]
        
        # Calculate quality score
        quality_score = len(present_fields) / len(mandatory_fields) if mandatory_fields else 0
        
        # Build validation report
        report_lines = [
            f"📋 **Validation Report**",
            f"",
            f"✅ **Present Fields ({len(present_fields)}/{len(mandatory_fields)}):**",
        ]
        
        for field in present_fields:
            report_lines.append(f"  • {field}: {found_fields[field]}")
        
        if missing_fields:
            report_lines.append(f"")
            report_lines.append(f"❌ **Missing Fields ({len(missing_fields)}):**")
            for field in missing_fields:
                report_lines.append(f"  • {field}")
        
        report_lines.append(f"")
        report_lines.append(f"📊 **Quality Score: {quality_score*100:.1f}%**")
        
        answer = "\n".join(report_lines)
        
        return ToolResponse(
            answer=answer,
            tool_name="validate_fields",
            structured_data={
                "present_fields": present_fields,
                "missing_fields": missing_fields,
                "quality_score": quality_score,
                "validation_status": "complete" if not missing_fields else "incomplete"
            },
            confidence=0.95
        )
    
    @staticmethod
    def cross_doc_search(query: str, hit_chunks: List[Dict], 
                        all_chunks: List[Dict]) -> ToolResponse:
        """
        Compare data across multiple documents.
        Tool for COMPARISON intent.
        """
        chunks = all_chunks if all_chunks else hit_chunks
        
        if not chunks or len(chunks) < 2:
            return ToolResponse(
                "Need at least 2 documents to compare. Currently have: 1 document",
                "cross_doc_search",
                confidence=0.5
            )
        
        # Group chunks by document
        docs_data = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "unknown")
            if doc_id not in docs_data:
                docs_data[doc_id] = {
                    "chunks": [],
                    "fields": {}
                }
            docs_data[doc_id]["chunks"].append(chunk)
            docs_data[doc_id]["fields"].update(chunk.get("extracted_fields", {}))
        
        # Build comparison report
        comparison_lines = ["📊 **Document Comparison**", ""]
        
        # Extract common and unique fields
        all_field_keys = set()
        for doc_data in docs_data.values():
            all_field_keys.update(doc_data["fields"].keys())
        
        # Create comparison table
        comparison_lines.append("| Field | " + " | ".join(docs_data.keys()) + " |")
        comparison_lines.append("|-------|" + "|".join(["---"] * len(docs_data)) + "|")
        
        for field in sorted(all_field_keys):
            row = f"| {field} |"
            for doc_id in docs_data.keys():
                value = docs_data[doc_id]["fields"].get(field, "❌ Missing")
                row += f" {value} |"
            comparison_lines.append(row)
        
        answer = "\n".join(comparison_lines)
        
        return ToolResponse(
            answer=answer,
            tool_name="cross_doc_search",
            structured_data={
                "document_count": len(docs_data),
                "documents": {
                    doc_id: {
                        "chunk_count": len(data["chunks"]),
                        "fields": data["fields"]
                    }
                    for doc_id, data in docs_data.items()
                }
            },
            confidence=0.80
        )
    
    @staticmethod
    def risk_analysis(query: str, hit_chunks: List[Dict], 
                     all_chunks: List[Dict]) -> ToolResponse:
        """
        Detect anomalies and flag risks.
        Tool for RISK intent.
        """
        chunks = all_chunks if all_chunks else hit_chunks
        
        if not chunks:
            return ToolResponse("No data to analyze for risks", "risk_analysis", confidence=0.2)
        
        # Aggregate fields
        all_fields = {}
        for chunk in chunks:
            all_fields.update(chunk.get("extracted_fields", {}))
        
        # Detect risks
        risks = []
        
        # Check for missing critical fields
        critical_fields = ["policy_number", "claim_amount", "incident_date", "bank_account"]
        missing_critical = [f for f in critical_fields if f not in all_fields]
        if missing_critical:
            risks.append({
                "flag": "Missing Critical Fields",
                "severity": "high",
                "description": f"Missing: {', '.join(missing_critical)}",
                "recommendation": "Request missing information from claimant"
            })
        
        # Check for unusual amounts (simple heuristic)
        try:
            if "claim_amount" in all_fields:
                amount = float(all_fields["claim_amount"])
                if amount > 1000000:
                    risks.append({
                        "flag": "Unusually High Claim Amount",
                        "severity": "medium",
                        "description": f"Claim amount (${amount:,.0f}) exceeds typical threshold",
                        "recommendation": "Flag for manual review by senior claims officer"
                    })
                elif amount <= 0:
                    risks.append({
                        "flag": "Invalid Amount",
                        "severity": "high",
                        "description": "Claim amount is zero or negative",
                        "recommendation": "Request clarification on claim amount"
                    })
        except (ValueError, TypeError):
            pass
        
        # Check for incomplete data
        if len(all_fields) < 3:
            risks.append({
                "flag": "Incomplete Data",
                "severity": "medium",
                "description": "Very few fields extracted from document",
                "recommendation": "Verify document quality or request additional documents"
            })
        
        # Check for date logic
        try:
            if "incident_date" in all_fields:
                # Simple check: date should be in past
                import datetime
                date_str = all_fields["incident_date"]
                # This is simplified; real implementation would parse various date formats
                if "future" in date_str.lower() or "2027" in date_str or "2028" in date_str:
                    risks.append({
                        "flag": "Invalid Date",
                        "severity": "high",
                        "description": "Incident date appears to be in the future",
                        "recommendation": "Verify incident date with claimant"
                    })
        except:
            pass
        
        # Build risk report
        risk_lines = ["🚨 **Risk Analysis Report**", ""]
        
        if risks:
            for i, risk in enumerate(risks, 1):
                severity_emoji = "🔴" if risk["severity"] == "high" else "🟡"
                risk_lines.append(f"{severity_emoji} **{i}. {risk['flag']}** ({risk['severity'].upper()})")
                risk_lines.append(f"   Description: {risk['description']}")
                risk_lines.append(f"   Action: {risk['recommendation']}")
                risk_lines.append("")
        else:
            risk_lines.append("✅ No significant risks detected")
        
        answer = "\n".join(risk_lines)
        
        return ToolResponse(
            answer=answer,
            tool_name="risk_analysis",
            structured_data={
                "risk_count": len(risks),
                "high_severity": len([r for r in risks if r["severity"] == "high"]),
                "medium_severity": len([r for r in risks if r["severity"] == "medium"]),
                "risks": risks
            },
            confidence=0.75
        )
    
    @staticmethod
    def workflow_generate(query: str, hit_chunks: List[Dict], 
                         all_chunks: List[Dict]) -> ToolResponse:
        """
        Generate recommended next steps and workflow.
        Tool for WORKFLOW intent.
        """
        chunks = all_chunks if all_chunks else hit_chunks
        
        if not chunks:
            return ToolResponse("No documents to process", "workflow_generate", confidence=0.2)
        
        # Validate first
        validation_result = DocumentTools.validate_fields(query, hit_chunks, all_chunks)
        validation_data = validation_result.structured_data or {}
        
        # Analyze risks
        risk_result = DocumentTools.risk_analysis(query, hit_chunks, all_chunks)
        risk_data = risk_result.structured_data or {}
        
        # Build workflow based on validation and risk
        workflow_steps = []
        quality_score = validation_data.get("quality_score", 0)
        missing_fields = validation_data.get("missing_fields", [])
        high_risk_count = risk_data.get("high_severity", 0)
        
        # Step 1: Initial Review
        workflow_steps.append({
            "step": 1,
            "action": "Initial Document Review",
            "status": "completed",
            "duration": "5 min",
            "notes": f"Document quality score: {quality_score*100:.1f}%"
        })
        
        # Step 2: Data Completion
        if missing_fields:
            workflow_steps.append({
                "step": 2,
                "action": "Request Missing Information",
                "status": "pending",
                "duration": "24 hours",
                "missing_fields": missing_fields,
                "notes": f"Need to collect: {', '.join(missing_fields)}"
            })
        else:
            workflow_steps.append({
                "step": 2,
                "action": "Data Complete - Proceed to Verification",
                "status": "completed",
                "duration": "0 min"
            })
        
        # Step 3: Risk Assessment
        if high_risk_count > 0:
            workflow_steps.append({
                "step": 3,
                "action": "Flag for Senior Review",
                "status": "pending",
                "duration": "2-3 hours",
                "notes": f"{high_risk_count} high-risk flags detected"
            })
        else:
            workflow_steps.append({
                "step": 3,
                "action": "Risk Assessment Passed",
                "status": "completed"
            })
        
        # Step 4: Approval
        workflow_steps.append({
            "step": 4,
            "action": "Approval & Processing",
            "status": "pending" if high_risk_count == 0 and not missing_fields else "blocked",
            "duration": "1-2 hours" if high_risk_count == 0 and not missing_fields else "pending",
            "notes": "Ready for approval" if high_risk_count == 0 and not missing_fields else "Blocked - Complete earlier steps first"
        })
        
        # Build workflow report
        workflow_lines = ["📋 **Recommended Workflow**", ""]
        
        for step in workflow_steps:
            status_emoji = "✅" if step["status"] == "completed" else "⏳" if step["status"] == "pending" else "🚫"
            workflow_lines.append(f"{status_emoji} **Step {step['step']}: {step['action']}**")
            workflow_lines.append(f"   Status: {step['status'].upper()}")
            workflow_lines.append(f"   Duration: {step.get('duration', 'N/A')}")
            if "notes" in step:
                workflow_lines.append(f"   Notes: {step['notes']}")
            if "missing_fields" in step:
                workflow_lines.append(f"   Missing: {', '.join(step['missing_fields'])}")
            workflow_lines.append("")
        
        answer = "\n".join(workflow_lines)
        
        return ToolResponse(
            answer=answer,
            tool_name="workflow_generate",
            structured_data={
                "workflow_status": "blocked" if missing_fields or high_risk_count > 0 else "ready",
                "total_steps": len(workflow_steps),
                "steps": workflow_steps
            },
            confidence=0.80
        )

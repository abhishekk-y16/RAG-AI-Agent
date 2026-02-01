"""
Tool orchestrator - routes intents to appropriate tools and manages execution.
"""
from typing import List, Dict, Any, Tuple
from .intent_classifier import Intent
from .tools import DocumentTools, ToolResponse


class ToolOrchestrator:
    """Routes query intents to appropriate tools and executes them"""
    
    # Map intents to tool methods
    INTENT_TOOL_MAP = {
        Intent.FACT: DocumentTools.retrieve_specific_field,
        Intent.SUMMARY: DocumentTools.generate_summary,
        Intent.EXTRACTION: DocumentTools.extract_structured_data,
        Intent.VALIDATION: DocumentTools.validate_fields,
        Intent.COMPARISON: DocumentTools.cross_doc_search,
        Intent.RISK: DocumentTools.risk_analysis,
        Intent.WORKFLOW: DocumentTools.workflow_generate,
    }
    
    @staticmethod
    def execute(intent: Intent, query: str, hit_chunks: List[Dict], 
                all_chunks: List[Dict], **kwargs) -> ToolResponse:
        """
        Execute the appropriate tool for the given intent.
        
        Args:
            intent: Classified intent type
            query: User query text
            hit_chunks: Retrieved relevant chunks
            all_chunks: All available chunks
            **kwargs: Additional arguments (e.g., mandatory_fields for validation)
        
        Returns:
            ToolResponse with answer and structured data
        """
        
        # Get the tool function
        tool_func = ToolOrchestrator.INTENT_TOOL_MAP.get(intent)
        
        if not tool_func:
            return ToolResponse(
                "Unable to process this request type",
                "unknown",
                confidence=0.0
            )
        
        # Execute the tool
        try:
            if intent == Intent.VALIDATION:
                # Pass mandatory_fields for validation tool
                mandatory_fields = kwargs.get("mandatory_fields", None)
                return tool_func(query, hit_chunks, all_chunks, mandatory_fields)
            else:
                return tool_func(query, hit_chunks, all_chunks)
        except Exception as e:
            return ToolResponse(
                f"Error executing tool: {str(e)}",
                tool_func.__name__,
                confidence=0.0
            )
    
    @staticmethod
    def get_tool_description(intent: Intent) -> str:
        """Get human-readable description of what a tool does"""
        descriptions = {
            Intent.FACT: "Retrieve specific factual information from documents",
            Intent.SUMMARY: "Generate a summary of the document(s)",
            Intent.EXTRACTION: "Extract data in structured format (JSON/table)",
            Intent.VALIDATION: "Validate document completeness and data quality",
            Intent.COMPARISON: "Compare information across multiple documents",
            Intent.RISK: "Detect anomalies and flag potential risks",
            Intent.WORKFLOW: "Generate recommended next steps and workflow",
        }
        return descriptions.get(intent, "Unknown tool")

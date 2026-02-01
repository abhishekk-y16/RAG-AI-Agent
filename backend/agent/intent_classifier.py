"""
Intent classifier for RAG agent.
Classifies user queries to determine the appropriate tool/action.
"""
import re
from enum import Enum
from typing import Tuple


class Intent(Enum):
    """Query intent types"""
    FACT = "fact"                    # Direct factual questions
    SUMMARY = "summary"              # Summarization requests
    EXTRACTION = "extraction"        # Data extraction (JSON, CSV, etc.)
    VALIDATION = "validation"        # Completeness/consistency checks
    COMPARISON = "comparison"        # Cross-document comparison
    RISK = "risk"                    # Risk/anomaly detection
    WORKFLOW = "workflow"            # Workflow generation, next steps
    CLARIFICATION = "clarification"  # Ask user for more info


def classify_intent(query: str) -> Tuple[Intent, float]:
    """
    Classify query intent using rule-based heuristics.
    
    Args:
        query: User query text
    
    Returns:
        (intent, confidence_score)
    """
    query_lower = query.lower().strip()
    
    # Summary intent patterns
    summary_patterns = [
        r'\bsummarize\b', r'\bsummary\b', r'sum up', r'brief overview',
        r'give me an overview', r'what is', r'tell me about', r'explain',
        r'\bgist\b', r'main points', r'key information'
    ]
    if any(re.search(pattern, query_lower) for pattern in summary_patterns):
        return Intent.SUMMARY, 0.95
    
    # Extraction intent patterns
    extraction_patterns = [
        r'extract', r'pull out', r'give me.*field', r'in json', r'in csv',
        r'as table', r'list all', r'get all', r'what are the', r'provide.*details',
        r'structured.*format'
    ]
    if any(re.search(pattern, query_lower) for pattern in extraction_patterns):
        return Intent.EXTRACTION, 0.92
    
    # Validation intent patterns
    validation_patterns = [
        r'missing', r'incomplete', r'check if', r'verify', r'validate',
        r'is.*present', r'do.*have', r'required', r'mandatory', r'what\'s missing',
        r'is everything', r'all.*filled', r'complete'
    ]
    if any(re.search(pattern, query_lower) for pattern in validation_patterns):
        return Intent.VALIDATION, 0.90
    
    # Comparison intent patterns
    comparison_patterns = [
        r'compare', r'difference', r'between', r'versus', r'vs\.?', r'similar',
        r'both.*documents', r'multiple.*documents', r'all.*claims', r'across'
    ]
    if any(re.search(pattern, query_lower) for pattern in comparison_patterns):
        # Only if multiple docs likely
        if 'all' in query_lower or 'both' in query_lower or 'across' in query_lower:
            return Intent.COMPARISON, 0.88
    
    # Risk/Anomaly intent patterns
    risk_patterns = [
        r'risk', r'red flag', r'suspicious', r'anomal', r'unusual', r'wrong',
        r'inconsist', r'duplicate', r'mismatch', r'concern', r'issue', r'problem',
        r'verify authenticity', r'check for fraud'
    ]
    if any(re.search(pattern, query_lower) for pattern in risk_patterns):
        return Intent.RISK, 0.89
    
    # Workflow intent patterns
    workflow_patterns = [
        r'next.*step', r'what should', r'what.*do', r'process', r'workflow',
        r'how.*proceed', r'what.*follow', r'checklist', r'action.*plan',
        r'recommendation', r'suggest'
    ]
    if any(re.search(pattern, query_lower) for pattern in workflow_patterns):
        return Intent.WORKFLOW, 0.87
    
    # Default to FACT (factual question)
    return Intent.FACT, 0.70


def get_tool_for_intent(intent: Intent) -> str:
    """Map intent to appropriate tool/handler"""
    tool_map = {
        Intent.FACT: "retrieve_specific_field",
        Intent.SUMMARY: "generate_summary",
        Intent.EXTRACTION: "extract_structured_data",
        Intent.VALIDATION: "validate_fields",
        Intent.COMPARISON: "cross_doc_search",
        Intent.RISK: "risk_analysis",
        Intent.WORKFLOW: "workflow_generate",
        Intent.CLARIFICATION: "ask_clarification"
    }
    return tool_map.get(intent, "retrieve_specific_field")

"""
Extract structured fields from document chunks.
Focuses on common document types: insurance claims, forms, agreements.
"""
import re
from typing import Dict, Any, List


def extract_fields_from_text(text: str) -> Dict[str, Any]:
    """
    Extract structured fields from document text.
    Covers: policy numbers, amounts, dates, names, contact info, bank details.
    
    Returns:
    {
        "policy_number": "...",
        "claim_amount": 50000.0,
        "incident_date": "2026-01-15",
        "policy_holder": "...",
        "claimant_name": "...",
        "bank_account": "...",
        "ifsc_code": "...",
        "phone": "...",
        "email": "...",
    }
    """
    fields = {}
    
    # Policy/Claim Number patterns
    policy_match = re.search(
        r'(?:policy\s*(?:no|number|#)|policy\s*id|claim\s*(?:no|number|#)|claim\s*id)\s*[:\-]?\s*([A-Z0-9\-]{6,})',
        text,
        re.IGNORECASE
    )
    if policy_match:
        fields["policy_number"] = policy_match.group(1).strip()
    
    # Claim/Insurance Amount patterns
    amount_match = re.search(
        r'(?:claim\s*amount|amount\s*claimed|sum\s*(?:assured|insured)|coverage\s*amount|premium|amount)\s*[:\-]?\s*(?:₹|rs|inr|usd|\$)?\s*([\d,]+(?:\.\d{2})?)',
        text,
        re.IGNORECASE
    )
    if amount_match:
        try:
            amount_str = amount_match.group(1).replace(",", "")
            fields["claim_amount"] = float(amount_str)
        except:
            pass
    
    # Date patterns (DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY, DD.MM.YYYY)
    date_match = re.search(
        r'(?:incident\s*date|date\s*(?:of|off)\s*incident|dob|date\s*of\s*birth|submission\s*date|date\s*of\s*claim)\s*[:\-]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})',
        text,
        re.IGNORECASE
    )
    if date_match:
        fields["incident_date"] = date_match.group(1).strip()
    
    # Names (Policy Holder, Claimant, etc.)
    name_match = re.search(
        r'(?:policy\s*(?:holder|owner)|insured\s*person|claimant|patient\s*name)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        text,
        re.IGNORECASE
    )
    if name_match:
        fields["policy_holder"] = name_match.group(1).strip()
    
    # Bank Account Number (10-18 digits)
    account_match = re.search(
        r'(?:account\s*(?:no|number|#)|bank\s*account)\s*[:\-]?\s*(\d{10,18})',
        text,
        re.IGNORECASE
    )
    if account_match:
        fields["bank_account"] = account_match.group(1).strip()
    
    # IFSC Code (4 letters + 0 + 6 alphanumeric)
    ifsc_match = re.search(
        r'(?:ifsc|ifsc\s*code)\s*[:\-]?\s*([A-Z]{4}0[A-Z0-9]{6})',
        text,
        re.IGNORECASE
    )
    if ifsc_match:
        fields["ifsc_code"] = ifsc_match.group(1).strip()
    
    # Phone Number (10 digits, with optional +91, spaces, dashes)
    phone_match = re.search(
        r'(?:phone|mobile|contact|phone\s*number)\s*[:\-]?\s*(?:\+91[-.\s]?)?\d{10}',
        text,
        re.IGNORECASE
    )
    if phone_match:
        # Extract just the digits
        phone_digits = re.findall(r'\d', phone_match.group(0))
        if len(phone_digits) >= 10:
            fields["phone"] = "".join(phone_digits[-10:])
    
    # Email
    email_match = re.search(
        r'(?:email|e\-?mail|email\s*id)\s*[:\-]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        text,
        re.IGNORECASE
    )
    if email_match:
        fields["email"] = email_match.group(1).strip()
    
    return fields


def get_mandatory_fields_for_type(doc_type: str) -> List[str]:
    """
    Return list of mandatory fields based on document type.
    doc_type: "insurance_claim", "health_claim", "form", "agreement", etc.
    """
    mandatory_fields = {
        "insurance_claim": [
            "policy_number",
            "claim_amount",
            "incident_date",
            "policy_holder",
            "bank_account",
            "ifsc_code"
        ],
        "health_claim": [
            "policy_number",
            "claim_amount",
            "incident_date",
            "policy_holder",
            "bank_account",
            "ifsc_code"
        ],
        "form": [
            "policy_holder",
            "contact",
            "email"
        ],
        "agreement": [
            "policy_number",
        ],
    }
    
    return mandatory_fields.get(doc_type, [])

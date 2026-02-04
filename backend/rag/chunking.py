import tiktoken
from .field_extractor import extract_fields_from_text


def chunk_text(text: str, chunk_tokens: int = 450, overlap_tokens: int = 80):
    """Chunk text into voerlapping segments with Field extraction"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    position = 0
    
    while start < len(tokens):
        end = start + chunk_tokens
        chunk_content = enc.decode(tokens[start:end])
        
        # Extract structured fields from chunk
        extracted_fields = extract_fields_from_text(chunk_content)
        
        # Store chunk with metadata
        chunk_obj = {
            "text": chunk_content,
            "start_token": start,
            "end_token": end,
            "token_count": len(tokens[start:end]),
            "position": position,
            "extracted_fields": extracted_fields
        }
        chunks.append(chunk_obj)
        
        start = end - overlap_tokens
        if start < 0:
            start = 0
        position += 1

    return chunks

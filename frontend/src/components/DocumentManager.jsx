import React, { useState, useRef } from "react";
import "../styles/DocumentManager.css";

function DocumentManager({ onDocumentUpload, documents = [] }) {
  const [uploading, setUploading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);

  const formatDate = (dateString) => {
    if (!dateString) return new Date().toLocaleDateString();
    try {
      const date = new Date(dateString);
      return isNaN(date.getTime()) ? new Date().toLocaleDateString() : date.toLocaleDateString();
    } catch {
      return new Date().toLocaleDateString();
    }
  };

  const normalizeDocument = (doc) => ({
    id: doc.id || doc.doc_id || Date.now(),
    name: doc.name || doc.filename || 'Unknown',
    pages: doc.pages || doc.num_pages || 0,
    chunks: doc.chunks || doc.num_chunks || 0,
    uploadedAt: doc.uploadedAt || doc.uploaded_at || new Date().toISOString(),
  });

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      console.log("Uploading file:", file.name);
      const response = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      
      console.log("Upload response status:", response.status);
      const data = await response.json();
      console.log("Upload response data:", data);
      
      if (data.error) {
        setUploadError(data.error);
        console.error("Upload error:", data.error);
      } else if (data.status === "ok" || data.document_id) {
        const normalizedDoc = normalizeDocument(data);
        console.log("Normalized document:", normalizedDoc);
        onDocumentUpload(normalizedDoc);
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      } else {
        setUploadError("Unexpected response format");
        console.error("Unexpected response:", data);
      }
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(error.message || "Network error - make sure backend is running on port 8000");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="document-manager">
      <h2>📄 Document Manager</h2>
      
      <div className="upload-section">
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleUpload}
          disabled={uploading}
          accept=".pdf,.txt"
          style={{ display: "none" }}
        />
        <button
          className="btn btn-primary"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? "⏳ Uploading..." : "📤 Upload Document"}
        </button>
      </div>

      {uploadError && (
        <div className="error-box">
          ❌ {uploadError}
        </div>
      )}

      {documents.length > 0 && (
        <div className="documents-list">
          <h3>Uploaded Documents ({documents.length})</h3>
          <div className="doc-items">
            {documents.map((doc, idx) => (
              <div
                key={doc.id || idx}
                className={`doc-item ${selectedDoc?.id === doc.id ? "selected" : ""}`}
                onClick={() => setSelectedDoc(doc)}
              >
                <div className="doc-icon">📄</div>
                <div className="doc-info">
                  <h4>{doc.name}</h4>
                  <p>{doc.pages || 0} pages • {doc.chunks || 0} chunks</p>
                  <small>{formatDate(doc.uploadedAt)}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedDoc && (
        <div className="doc-details">
          <h3>Document Details</h3>
          <div className="details-grid">
            <div className="detail">
              <label>Name:</label>
              <span>{selectedDoc.name}</span>
            </div>
            <div className="detail">
              <label>Pages:</label>
              <span>{selectedDoc.pages}</span>
            </div>
            <div className="detail">
              <label>Chunks:</label>
              <span>{selectedDoc.chunks}</span>
            </div>
            <div className="detail">
              <label>Uploaded:</label>
              <span>{formatDate(selectedDoc.uploadedAt)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default DocumentManager;

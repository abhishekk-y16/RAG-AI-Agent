import { useState, useRef, useEffect } from "react";
import "./ChatWidget.css";

const QUICK_ACTIONS = [
  { id: 1, label: "Summarize Document", icon: "📋" },
  { id: 2, label: "Extract Key Details", icon: "🔍" },
  { id: 3, label: "Check Missing Info", icon: "⚠️" },
  { id: 4, label: "Convert to JSON", icon: "📊" },
];

const SUGGESTED_QUESTIONS = [
  "What is this document about?",
  "Extract key entities from the document",
  "Are any fields missing?",
  "Summarize in simple terms",
];

export default function ChatWidget({ uploadedDocument, documents = [], onDocumentUpload }) {
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [interactionCount, setInteractionCount] = useState(0);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [msgs, loading]);

  const handleQuickAction = (action) => {
    const promptMap = {
      1: "Please summarize the entire document in clear, concise points.",
      2: "Extract and list all key details, entities, and important information from the document.",
      3: "Identify any missing or incomplete information in the document.",
      4: "Extract all key information and structure it in JSON format.",
    };
    setText(promptMap[action.id]);
  };

  const handleSuggestedQuestion = (question) => {
    setText(question);
  };

  async function send() {
    const msg = text.trim();
    if (!msg || !documentInfo) return;

    setMsgs((m) => [...m, { role: "user", text: msg }]);
    setText("");
    setInteractionCount((c) => c + 1);
    if (interactionCount >= 2) {
      setShowSuggestions(false);
    }
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });

      const data = await res.json();
      
      const response = {
        role: "bot",
        text: data.answer,
        intent: data.intent || null,
        sources: data.sources || [],
        confidence: data.confidence || null,
        notFound: data.not_found || false,
      };

      setMsgs((m) => [...m, response]);
    } catch (error) {
      setMsgs((m) => [
        ...m,
        {
          role: "bot",
          text: "Unable to process your question. Make sure the backend is running on http://localhost:8000",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    const supportedTypes = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"];
    if (!supportedTypes.includes(file.type)) {
      setMsgs((m) => [
        ...m,
        {
          role: "bot",
          text: "⚠️ Unsupported format. Please upload PDF, DOCX, or TXT.",
          error: true,
        },
      ]);
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setMsgs((m) => [
        ...m,
        {
          role: "bot",
          text: "⚠️ File size exceeds 10MB limit",
          error: true,
        },
      ]);
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (data.error) {
        setMsgs((m) => [
          ...m,
          {
            role: "bot",
            text: `❌ Error processing document: ${data.error}`,
            error: true,
          },
        ]);
      } else {
        // Show validation warnings if fields are missing
        let validationMsg = "";
        if (data.missing_fields && data.missing_fields.length > 0) {
          validationMsg = `⚠️ Missing fields: ${data.missing_fields.join(", ")}`;
        }
        
        const docInfo = {
          name: file.name,
          pages: data.pages || "Unknown",
          chunks: data.chunks || 0,
          uploadedAt: new Date(data.uploadedAt || new Date()),
          document_id: data.document_id,
          quality_score: data.quality_score,
          missing_fields: data.missing_fields || []
        };
        setDocumentInfo(docInfo);
        onDocumentUpload(docInfo);
        setShowSuggestions(true);
        setInteractionCount(0);
        setMsgs([]);
        
        // Show validation message if needed
        if (validationMsg) {
          setMsgs([{
            role: "bot",
            text: validationMsg,
            notFound: true  // Use amber styling for warnings
          }]);
        }
      }
    } catch (error) {
      setMsgs((m) => [
        ...m,
        {
          role: "bot",
          text: "❌ Error uploading file. Ensure the backend is running.",
          error: true,
        },
      ]);
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  function getTimeAgo(date) {
    if (!date) return "";
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    
    // Show full date/time for older than 2 minutes
    if (minutes >= 2) {
      const dateStr = date.toLocaleDateString();
      const timeStr = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return `${dateStr} ${timeStr}`;
    }
    
    if (minutes < 1) return "just now";
    if (minutes === 1) return "1 min ago";
    return `${minutes} min ago`;
  }

  return (
    <div className="chat-widget">
      {!documentInfo ? (
        <div className="empty-state">
          <div className="empty-state-icon">📄</div>
          <h2 className="empty-state-title">Upload a document to get started</h2>
          <p className="empty-state-subtitle">
            This assistant answers questions strictly from your uploaded document.
          </p>
          
          <div className="upload-zone">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={handleFileUpload}
              style={{ display: "none" }}
              disabled={uploading || loading}
            />
            <button
              className="upload-btn-primary"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Upload Document"}
            </button>
          </div>

          <p className="upload-helper">
            Supported formats: PDF, DOCX, TXT · Max 10MB
          </p>
        </div>
      ) : (
        <>
          {/* Multi-Document Selector */}
          {documents.length > 0 && (
            <div className="multi-doc-selector">
              <h3>📚 Available Documents ({documents.length})</h3>
              <div className="doc-selector-list">
                {documents.map((doc, idx) => {
                  const docKey = doc.id || idx;
                  return (
                    <button
                      key={docKey}
                      className={`doc-selector-btn ${selectedDocId === docKey ? "active" : ""}`}
                      onClick={() => {
                        console.log("Selected document:", doc);
                        setSelectedDocId(docKey);
                        setDocumentInfo(doc);
                        setMsgs([]);
                        setShowSuggestions(true);
                        setInteractionCount(0);
                      }}
                    >
                      <span className="selector-icon">📄</span>
                      <span className="selector-text">{doc.name}</span>
                      <span className="selector-meta">({doc.pages || 0} pages)</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {documentInfo && (
            <>
              <div className="context-lock-header">
                <div className="document-badge">
                  <span className="doc-icon">📄</span>
                  <div className="doc-info">
                    <span className="doc-name">{documentInfo.name}</span>
                    <span className="doc-meta">
                      Pages: {documentInfo.pages || 0} · {getTimeAgo(documentInfo.uploadedAt)}
                    </span>
                  </div>
                </div>
                <button
                  className="change-doc-btn"
                  onClick={() => {
                    setDocumentInfo(null);
                    setMsgs([]);
                    setShowSuggestions(false);
                    setInteractionCount(0);
                  }}
                >
                  Change Document
                </button>
              </div>

              <div className="main-layout">
            <div className="left-panel">
              <div className="info-card">
                <h3 className="card-title">📊 Document Summary</h3>
                <div className="doc-details">
                  <div className="detail-row">
                    <span className="label">File Name:</span>
                    <span className="value">{documentInfo.name}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Pages:</span>
                    <span className="value">{documentInfo.pages}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Uploaded:</span>
                    <span className="value">{getTimeAgo(documentInfo.uploadedAt)}</span>
                  </div>
                </div>
              </div>

              <div className="quick-actions-card">
                <h3 className="card-title">⚡ Quick Actions</h3>
                <div className="actions-grid">
                  {QUICK_ACTIONS.map((action) => (
                    <button
                      key={action.id}
                      className="action-btn"
                      onClick={() => handleQuickAction(action)}
                    >
                      <span className="action-icon">{action.icon}</span>
                      <span className="action-label">{action.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {showSuggestions && (
                <div className="suggested-card">
                  <h3 className="card-title">💡 Try Asking</h3>
                  <div className="suggested-list">
                    {SUGGESTED_QUESTIONS.map((q, i) => (
                      <button
                        key={i}
                        className="suggested-btn"
                        onClick={() => handleSuggestedQuestion(q)}
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="right-panel">
              <div className="chat-messages">
                {msgs.length === 0 && (
                  <div className="chat-welcome">
                    <p>Start asking questions about your document!</p>
                  </div>
                )}

                {msgs.map((m, i) => (
                  <div key={i} className={`message-group ${m.role}`}>
                    {m.role === "user" ? (
                      <div className="message-user">
                        <div className="message-content">{m.text}</div>
                      </div>
                    ) : (
                      <div className="message-bot">
                        {m.error ? (
                          <div className="message-content error-message">
                            <span className="error-icon">❌</span>
                            {m.text}
                          </div>
                        ) : m.notFound ? (
                          <div className="message-box not-found">
                            <div className="not-found-header">
                              <span className="warning-icon">⚠️</span>
                              <span>Information not found</span>
                            </div>
                            <p className="not-found-text">{m.text}</p>
                            <p className="not-found-hint">
                              💡 Tip: Try asking about details that appear in the document.
                            </p>
                          </div>
                        ) : (
                          <div className="message-box">
                            <div className="message-content">{m.text}</div>
                            {(m.sources || m.confidence || m.intent) && (
                              <div className="message-footer">
                                {m.intent && (
                                  <span className="intent-badge">
                                    🎯 Intent: {m.intent}
                                  </span>
                                )}
                                {m.sources && m.sources.length > 0 && (
                                  <span className="source-badge">
                                    📌 {m.sources.length} source(s)
                                  </span>
                                )}
                                {m.confidence && (
                                  <span className="confidence-badge">
                                    Confidence: {(m.confidence * 100).toFixed(0)}%
                                  </span>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}

                {loading && (
                  <div className="message-group bot">
                    <div className="message-bot">
                      <div className="message-box loading">
                        <div className="skeleton-line short"></div>
                        <div className="skeleton-line"></div>
                        <div className="skeleton-line medium"></div>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              <div className="chat-input-area">
                <input
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && !loading && send()}
                  placeholder="Ask a question about the uploaded document…"
                  disabled={loading || uploading}
                  className="chat-input"
                />
                <button
                  onClick={send}
                  disabled={loading || uploading || !text.trim()}
                  className="send-btn"
                >
                  {loading ? "..." : "Send"}
                </button>
              </div>
            </div>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}

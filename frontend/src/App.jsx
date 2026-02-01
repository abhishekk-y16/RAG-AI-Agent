import React, { useState } from "react";
import ChatWidget from "./ChatWidget";
import DocumentManager from "./components/DocumentManager";
import QuickTools from "./components/QuickTools";
import WorkflowExecutor from "./components/WorkflowExecutor";
import Analytics from "./components/Analytics";
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState("chat");
  const [uploadedDocument, setUploadedDocument] = useState(null);
  const [documents, setDocuments] = useState([]);

  const handleDocumentUpload = (doc) => {
    setUploadedDocument(doc);
    setDocuments([...documents, doc]);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <h1>🤖 RAG AI Agent</h1>
          <p className="subtitle">Document Intelligence Platform</p>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="app-nav">
        <button
          className={`nav-btn ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          💬 Chat & Q&A
        </button>
        <button
          className={`nav-btn ${activeTab === "documents" ? "active" : ""}`}
          onClick={() => setActiveTab("documents")}
        >
          📄 Documents
        </button>
        <button
          className={`nav-btn ${activeTab === "tools" ? "active" : ""}`}
          onClick={() => setActiveTab("tools")}
        >
          ⚡ Quick Tools
        </button>
        <button
          className={`nav-btn ${activeTab === "workflows" ? "active" : ""}`}
          onClick={() => setActiveTab("workflows")}
        >
          🔄 Workflows
        </button>
        <button
          className={`nav-btn ${activeTab === "analytics" ? "active" : ""}`}
          onClick={() => setActiveTab("analytics")}
        >
          📊 Analytics
        </button>
      </nav>

      {/* Main Content */}
      <main className="app-main">
        {activeTab === "chat" && (
          <ChatWidget
            uploadedDocument={uploadedDocument}
            documents={documents}
            onDocumentUpload={handleDocumentUpload}
          />
        )}

        {activeTab === "documents" && (
          <DocumentManager
            documents={documents}
            onDocumentUpload={handleDocumentUpload}
          />
        )}

        {activeTab === "tools" && <QuickTools documents={documents} />}

        {activeTab === "workflows" && <WorkflowExecutor documents={documents} />}

        {activeTab === "analytics" && <Analytics />}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>RAG AI Agent v4.0 • Phase 5 Complete • Production Ready</p>
      </footer>
    </div>
  );
}

export default App;

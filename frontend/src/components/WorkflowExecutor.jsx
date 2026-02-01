import React, { useState, useEffect } from "react";
import "../styles/WorkflowExecutor.css";

function WorkflowExecutor({ documents = [] }) {
  const [workflows, setWorkflows] = useState([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState(null);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [inputs, setInputs] = useState({});
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const fetchWorkflows = async () => {
    try {
      const response = await fetch("http://localhost:8000/workflows/available");
      const data = await response.json();
      setWorkflows(Object.keys(data.workflows || {}));
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch workflows:", error);
      setLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedWorkflow) return;

    setExecuting(true);
    try {
      const payload = {
        ...inputs,
        ...(selectedDoc && { document_id: selectedDoc }),
      };
      const response = await fetch(
        `http://localhost:8000/workflows/${selectedWorkflow}/execute`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Workflow execution failed:", error);
      setResult({ error: error.message });
    } finally {
      setExecuting(false);
    }
  };

  if (loading) return <div className="workflow-executor">⏳ Loading workflows...</div>;

  return (
    <div className="workflow-executor">
      <h2>🔄 Workflow Executor</h2>

      {/* Document Selector */}
      {documents.length > 0 && (
        <div className="workflow-doc-selector">
          <label>📄 Select Document (Optional):</label>
          <select
            value={selectedDoc || ""}
            onChange={(e) => setSelectedDoc(e.target.value || null)}
          >
            <option value="">None - Use provided text</option>
            {documents.map((doc, idx) => (
              <option key={doc.id || idx} value={doc.id || idx}>
                {doc.name} ({doc.pages} pages)
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="workflow-selection">
        <h3>Available Workflows</h3>
        <div className="workflow-list">
          {workflows.map((wf) => (
            <button
              key={wf}
              className={`workflow-btn ${selectedWorkflow === wf ? "active" : ""}`}
              onClick={() => {
                setSelectedWorkflow(wf);
                setInputs({});
                setResult(null);
              }}
            >
              {wf}
            </button>
          ))}
        </div>
      </div>

      {selectedWorkflow && (
        <div className="workflow-input">
          <h3>Input Parameters</h3>
          <div className="input-fields">
            <div className="input-field">
              <label>Text Content</label>
              <textarea
                value={inputs.text || ""}
                onChange={(e) => setInputs({ ...inputs, text: e.target.value })}
                placeholder="Enter text to process..."
                rows={6}
              />
            </div>
            <div className="input-field">
              <label>Document Type (optional)</label>
              <select
                value={inputs.document_type || ""}
                onChange={(e) =>
                  setInputs({ ...inputs, document_type: e.target.value })
                }
              >
                <option value="">Auto-detect</option>
                <option value="insurance_claim">Insurance Claim</option>
                <option value="contract">Contract</option>
                <option value="invoice">Invoice</option>
                <option value="report">Report</option>
              </select>
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleExecute}
            disabled={executing || !inputs.text}
          >
            {executing ? "⏳ Executing..." : "▶️ Execute Workflow"}
          </button>
        </div>
      )}

      {result && (
        <div className="workflow-result">
          <h3>Execution Result</h3>
          {result.error ? (
            <div className="error-box">❌ Error: {result.error}</div>
          ) : (
            <div className="result-box">
              <div className="result-status">
                <span className={`status ${result.status}`}>
                  {result.status?.toUpperCase()}
                </span>
                <span className="duration">⏱️ {result.duration_ms}ms</span>
              </div>

              {result.results && (
                <div className="results-detail">
                  <h4>Step Results:</h4>
                  <div className="steps-grid">
                    {Object.entries(result.results).map(([step, data]) => (
                      <div key={step} className="step-result">
                        <h5>📍 {step}</h5>
                        <pre>{JSON.stringify(data, null, 2)}</pre>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default WorkflowExecutor;

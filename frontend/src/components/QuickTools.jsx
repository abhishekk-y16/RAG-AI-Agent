import React, { useState } from "react";
import "../styles/QuickTools.css";

function QuickTools({ documents = [], selectedDocId = null }) {
  const [activeTab, setActiveTab] = useState("extract");
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState(selectedDocId);

  const tools = {
    extract: {
      name: "Field Extraction",
      icon: "🔍",
      endpoint: "/workflows/quick-extract",
      fields: [
        { name: "text", type: "textarea", label: "Text to Extract From", required: true },
        {
          name: "fields",
          type: "select-multiple",
          label: "Fields to Extract",
          options: ["email", "phone", "date", "amount", "url", "zipcode"],
        },
      ],
    },
    validate: {
      name: "Data Validation",
      icon: "✅",
      endpoint: "/workflows/quick-validate",
      fields: [
        { name: "data", type: "json", label: "Data to Validate", required: true },
        { name: "rules", type: "json", label: "Validation Rules", required: true },
      ],
    },
    summarize: {
      name: "Text Summarization",
      icon: "📝",
      endpoint: "/workflows/quick-summarize",
      fields: [
        { name: "text", type: "textarea", label: "Text to Summarize", required: true },
        {
          name: "max_sentences",
          type: "number",
          label: "Max Sentences",
          default: 3,
        },
      ],
    },
    classify: {
      name: "Document Classification",
      icon: "📊",
      endpoint: "/workflows/quick-classify",
      fields: [
        { name: "text", type: "textarea", label: "Document Text", required: true },
        {
          name: "categories",
          type: "select-multiple",
          label: "Categories",
          options: ["insurance_claim", "contract", "invoice", "report"],
        },
      ],
    },
  };

  const tool = tools[activeTab];

  const handleExecute = async () => {
    setLoading(true);
    try {
      const payload = {
        ...inputs,
        ...(selectedDoc && { document_id: selectedDoc }),
      };
      const response = await fetch(`http://localhost:8000${tool.endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const renderField = (field) => {
    const value = inputs[field.name] || field.default || "";

    if (field.type === "textarea") {
      return (
        <textarea
          key={field.name}
          value={value}
          onChange={(e) => setInputs({ ...inputs, [field.name]: e.target.value })}
          placeholder={field.label}
          rows={6}
        />
      );
    }

    if (field.type === "number") {
      return (
        <input
          key={field.name}
          type="number"
          value={value}
          onChange={(e) => setInputs({ ...inputs, [field.name]: parseInt(e.target.value) })}
          placeholder={field.label}
        />
      );
    }

    if (field.type === "json") {
      return (
        <textarea
          key={field.name}
          value={typeof value === "string" ? value : JSON.stringify(value, null, 2)}
          onChange={(e) => {
            try {
              setInputs({ ...inputs, [field.name]: JSON.parse(e.target.value) });
            } catch {
              setInputs({ ...inputs, [field.name]: e.target.value });
            }
          }}
          placeholder={`Enter JSON for ${field.label}`}
          rows={6}
          className="json-input"
        />
      );
    }

    if (field.type === "select-multiple") {
      return (
        <div key={field.name} className="select-multiple">
          {field.options.map((opt) => (
            <label key={opt}>
              <input
                type="checkbox"
                checked={(inputs[field.name] || []).includes(opt)}
                onChange={(e) => {
                  const arr = inputs[field.name] || [];
                  setInputs({
                    ...inputs,
                    [field.name]: e.target.checked
                      ? [...arr, opt]
                      : arr.filter((x) => x !== opt),
                  });
                }}
              />
              {opt}
            </label>
          ))}
        </div>
      );
    }

    return null;
  };

  return (
    <div className="quick-tools">
      <h2>⚡ Quick Tools</h2>

      {/* Document Selector */}
      {documents.length > 0 && (
        <div className="tool-doc-selector">
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

      <div className="tool-tabs">
        {Object.entries(tools).map(([key, t]) => (
          <button
            key={key}
            className={`tool-tab ${activeTab === key ? "active" : ""}`}
            onClick={() => {
              setActiveTab(key);
              setInputs({});
              setResult(null);
            }}
          >
            {t.icon} {t.name}
          </button>
        ))}
      </div>

      <div className="tool-content">
        <h3>{tool.icon} {tool.name}</h3>

        <div className="tool-inputs">
          {tool.fields.map((field) => (
            <div key={field.name} className="input-group">
              <label htmlFor={field.name}>{field.label}</label>
              {renderField(field)}
            </div>
          ))}
        </div>

        <button
          className="btn btn-primary"
          onClick={handleExecute}
          disabled={loading}
        >
          {loading ? "⏳ Processing..." : "▶️ Execute"}
        </button>
      </div>

      {result && (
        <div className="tool-result">
          <h3>Result</h3>
          {result.error ? (
            <div className="error-box">❌ Error: {result.error}</div>
          ) : (
            <div className="result-display">
              {result.extraction_result && (
                <div className="result-section">
                  <h4>Extracted Data</h4>
                  <pre>{JSON.stringify(result.extraction_result, null, 2)}</pre>
                </div>
              )}

              {result.validation_result && (
                <div className="result-section">
                  <h4>Validation Result</h4>
                  <div className={`validation-status ${result.validation_result.valid ? "valid" : "invalid"}`}>
                    {result.validation_result.valid ? "✅ Valid" : "❌ Invalid"}
                  </div>
                  {result.validation_result.errors?.length > 0 && (
                    <div className="errors">
                      <h5>Errors:</h5>
                      <ul>
                        {result.validation_result.errors.map((e, i) => (
                          <li key={i}>{e}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {result.summary_result && (
                <div className="result-section">
                  <h4>Summary</h4>
                  <p>{result.summary_result.summary}</p>
                  <small>Compression ratio: {(result.summary_result.compression_ratio * 100).toFixed(1)}%</small>
                </div>
              )}

              {result.classification_result && (
                <div className="result-section">
                  <h4>Classification</h4>
                  <div className="classification">
                    <span className="class">{result.classification_result.classification}</span>
                    <span className="confidence">
                      Confidence: {(result.classification_result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}

              {result.formatted && (
                <div className="result-section">
                  <h4>Formatted Output</h4>
                  <div className="format-tabs">
                    <button className="format-btn active">JSON</button>
                    <button className="format-btn">Markdown</button>
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

export default QuickTools;

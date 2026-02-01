import React, { useState, useEffect } from "react";
import "../styles/Analytics.css";

function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [queryHistory, setQueryHistory] = useState([]);
  const [risks, setRisks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
    try {
      const [analyticsRes, queryRes, risksRes] = await Promise.all([
        fetch("http://localhost:8000/analytics/system"),
        fetch("http://localhost:8000/audit-log/all?limit=10"),
        fetch("http://localhost:8000/risks/high-priority"),
      ]);

      const analyticsData = await analyticsRes.json();
      const queryData = await queryRes.json();
      const risksData = await risksRes.json();

      setAnalytics(analyticsData);
      setQueryHistory(queryData.events || []);
      setRisks(risksData.risks || []);
      setLoading(false);
    } catch (error) {
      console.error("Failed to fetch analytics:", error);
      setLoading(false);
    }
  };

  if (loading) return <div className="analytics">⏳ Loading analytics...</div>;

  return (
    <div className="analytics">
      <h2>📊 Analytics & Monitoring</h2>

      <div className="analytics-grid">
        {/* Summary Cards */}
        <div className="summary-cards">
          <div className="card">
            <div className="card-icon">📄</div>
            <div className="card-content">
              <h4>Total Documents</h4>
              <p className="metric">{analytics?.total_documents || 0}</p>
            </div>
          </div>

          <div className="card">
            <div className="card-icon">🔍</div>
            <div className="card-content">
              <h4>Total Queries</h4>
              <p className="metric">{analytics?.total_queries || 0}</p>
            </div>
          </div>

          <div className="card">
            <div className="card-icon">✅</div>
            <div className="card-content">
              <h4>Validations</h4>
              <p className="metric">{analytics?.total_validations || 0}</p>
            </div>
          </div>

          <div className="card">
            <div className="card-icon">⚠️</div>
            <div className="card-content">
              <h4>High Risks</h4>
              <p className="metric" style={{ color: risks.length > 0 ? "#e74c3c" : "#27ae60" }}>
                {risks.length}
              </p>
            </div>
          </div>
        </div>

        {/* Recent Queries */}
        <div className="section">
          <h3>🕐 Recent Activity</h3>
          <div className="query-history">
            {queryHistory.length === 0 ? (
              <p className="empty">No recent activity</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Type</th>
                    <th>Document</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {queryHistory.slice(0, 10).map((event, idx) => (
                    <tr key={idx}>
                      <td>
                        <span className="event-type">{event.event_type || "Query"}</span>
                      </td>
                      <td className="truncate">{event.metadata?.doc_id || "N/A"}</td>
                      <td className="time">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* High Priority Risks */}
        {risks.length > 0 && (
          <div className="section">
            <h3>⚠️ High Priority Risks</h3>
            <div className="risks-list">
              {risks.map((risk, idx) => (
                <div key={idx} className="risk-item">
                  <div className="risk-header">
                    <span className="risk-level">{risk.severity?.toUpperCase()}</span>
                    <span className="risk-doc">{risk.doc_id}</span>
                  </div>
                  <p className="risk-desc">{risk.risk_description}</p>
                  <small>{new Date(risk.flagged_at).toLocaleDateString()}</small>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <button className="btn btn-secondary" onClick={fetchAnalytics}>
        🔄 Refresh Analytics
      </button>
    </div>
  );
}

export default Analytics;

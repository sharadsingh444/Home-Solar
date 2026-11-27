import { useCallback, useEffect, useState } from 'react';
import { Activity, Database, Home, BrainCircuit, RefreshCw, AlertTriangle } from 'lucide-react';
import MetricCard from '../components/MetricCard.jsx';
import Panel from '../components/Panel.jsx';
import ModelComparisonChart from '../components/ModelComparisonChart.jsx';
import ImprovementChart from '../components/ImprovementChart.jsx';
import UpdatePanel from '../components/UpdatePanel.jsx';
import { api } from '../services/api.js';

function metric(value, digits = 2) {
  return value !== null && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

export default function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    setError('');
    try { setSummary(await api.dashboard()); }
    catch (err) { setError(err.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const models = summary?.models || {};
  const history = summary?.update_history || [];
  const latestRun = history.at(-1);
  const data = summary?.data || {};
  const eval24 = models.next_24h?.test_metrics || {};
  const eval30 = models.next_30d?.test_metrics || {};

  return (
    <div className="page-stack">
      <div className="page-heading">
        <div><p className="eyebrow">HOUSEHOLD ENERGY INTELLIGENCE</p><h1>Project dashboard</h1><p>Electricity forecasts, solar estimates and measurable model updates.</p></div>
        <button className="button button-light" onClick={refresh} disabled={loading}><RefreshCw size={16} /> Refresh</button>
      </div>
      {error && <div className="notice notice-error"><AlertTriangle size={16} /> {error}. Check that Express is running and the Python environment is configured.</div>}
      {loading && <div className="loading-line">Loading project status…</div>}
      <div className="metric-grid">
        <MetricCard label="Households in feature data" value={data.household_count ?? '—'} detail="Unique meter IDs" icon={Home} tone="green" />
        <MetricCard label="Processed feature rows" value={data.rows?.toLocaleString?.() ?? '—'} detail="Current feature dataset" icon={Database} tone="blue" />
        <MetricCard label="24-hour test MAE" value={metric(eval24.mae_kwh)} suffix=" kWh" detail={`${Number(eval24.rows || 0).toLocaleString()} test origins`} icon={Activity} tone="teal" />
        <MetricCard label="30-day test MAE" value={metric(eval30.mae_kwh)} suffix=" kWh" detail={`${Number(eval30.rows || 0).toLocaleString()} test origins`} icon={BrainCircuit} tone="purple" />
      </div>
      {data.warning && <div className="notice notice-warning"><AlertTriangle size={16} /> {data.warning} Latest feature timestamp: <strong>{data.latest_timestamp_utc || 'unknown'}</strong></div>}
      <div className="two-column-grid">
        <Panel title="Held-out model performance" subtitle="Current saved evaluation reports. MAE and RMSE are in kWh.">
          <ModelComparisonChart models={models} />
          <div className="mini-metrics-table">
            <div className="table-head"><span>Forecast</span><span>RMSE</span><span>WAPE</span><span>sMAPE</span></div>
            {[['24 hours', eval24], ['30 days', eval30]].map(([name, m]) => <div className="table-row" key={name}><strong>{name}</strong><span>{metric(m.rmse_kwh)}</span><span>{metric(m.wape_percent)}%</span><span>{metric(m.smape_percent)}%</span></div>)}
          </div>
        </Panel>
        <Panel title="Self-development history" subtitle="Existing model versus retrained candidate on the same new-data comparison window.">
          <ImprovementChart history={history} />
          {latestRun && <div className="latest-update-summary"><span className="status-pill good">Latest update: {latestRun.status}</span><small>{new Date(latestRun.completed_at_utc).toLocaleString()}</small></div>}
        </Panel>
      </div>
      <Panel title="Active model artifacts" subtitle="Models are reloaded automatically when a newly promoted artifact replaces the current file.">
        <div className="model-list">
          {Object.entries(models).map(([key, model]) => <div className="model-row" key={key}>
            <div><strong>{key === 'next_24h' ? '24-hour electricity model' : '30-day electricity model'}</strong><small>{model.selected_model || 'Model name not available'}</small></div>
            <span className={`status-pill ${model.available ? 'good' : 'muted'}`}>{model.available ? 'Artifact available' : 'Not trained'}</span>
          </div>)}
        </div>
      </Panel>
      <UpdatePanel onCompleted={refresh} />
    </div>
  );
}

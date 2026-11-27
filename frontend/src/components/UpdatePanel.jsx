import { useEffect, useState } from 'react';
import { Upload, RefreshCw, CheckCircle2, AlertCircle, LoaderCircle } from 'lucide-react';
import Panel from './Panel.jsx';
import { api } from '../services/api.js';

export default function UpdatePanel({ onCompleted }) {
  const [electricity, setElectricity] = useState(null);
  const [weather, setWeather] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!job?.id || !['queued', 'running'].includes(job.status)) return undefined;
    let active = true;
    const timer = setInterval(async () => {
      try {
        const next = await api.updateJob(job.id);
        if (!active) return;
        setJob(next);
        if (['completed', 'failed'].includes(next.status)) {
          setBusy(false);
          if (next.status === 'completed') onCompleted?.();
        }
      } catch (err) {
        if (active) { setError(err.message); setBusy(false); }
      }
    }, 2500);
    return () => { active = false; clearInterval(timer); };
  }, [job?.id, job?.status, onCompleted]);

  async function submit(event) {
    event.preventDefault();
    setError('');
    if (!electricity || !weather) { setError('Select both a new electricity CSV and its matching hourly weather CSV.'); return; }
    setBusy(true);
    setJob(null);
    try {
      const started = await api.startUpdate(electricity, weather);
      setJob(started);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  }

  const resultTargets = job?.result?.targets || {};
  return (
    <Panel title="Self-development: inject new data and retrain" subtitle="Upload a new meter batch and the matching hourly weather batch. HomeSolar reruns ETL, creates fresh targets, retrains candidate models and promotes only candidates that improve on newly arrived held-out data." className="update-panel">
      <form className="update-form" onSubmit={submit}>
        <label className="file-field">
          <span><Upload size={17} /> New electricity CSV</span>
          <input type="file" accept=".csv,text/csv" onChange={(e) => setElectricity(e.target.files?.[0] || null)} />
          <small>{electricity?.name || 'Expected same columns as the existing CEEW meter file'}</small>
        </label>
        <label className="file-field">
          <span><Upload size={17} /> Matching hourly weather CSV</span>
          <input type="file" accept=".csv,text/csv" onChange={(e) => setWeather(e.target.files?.[0] || null)} />
          <small>{weather?.name || 'Expected same columns and location IDs as the existing weather file'}</small>
        </label>
        <button className="button button-primary" type="submit" disabled={busy}>
          {busy ? <><LoaderCircle className="spin" size={17} /> Update running</> : <><RefreshCw size={17} /> Run ETL + retrain</>}
        </button>
      </form>
      <div className="notice notice-info">
        <strong>Promotion rule:</strong> the new candidate is compared with the current model on the same recent test rows. It replaces the deployed model only if MAE improves by at least 0.5%. Otherwise the current model is retained. Improvement is measured, never assumed.
      </div>
      {error && <div className="notice notice-error"><AlertCircle size={16} /> {error}</div>}
      {job && <div className={`job-status ${job.status}`}>
        <div className="job-title">
          {job.status === 'completed' ? <CheckCircle2 size={18} /> : job.status === 'failed' ? <AlertCircle size={18} /> : <LoaderCircle className="spin" size={18} />}
          <strong>Update job: {job.status}</strong><span>{job.stage}</span>
        </div>
        {job.error && <pre className="error-log">{job.error}</pre>}
        {job.result && <div className="update-result-grid">
          {Object.entries(resultTargets).map(([target, value]) => <div className="update-result" key={target}>
            <strong>{target === 'next_24h' ? '24-hour model' : '30-day model'}</strong>
            <span className={`status-pill ${value.status === 'promoted' ? 'good' : 'muted'}`}>{value.status}</span>
            {value.comparison && <p>Baseline MAE: {Number(value.comparison.baseline.mae_kwh).toFixed(3)} kWh<br />Candidate MAE: {Number(value.comparison.candidate.mae_kwh).toFixed(3)} kWh<br />Change: {Number(value.improvement_pct).toFixed(2)}%</p>}
            {value.reason && <p>{value.reason}</p>}
          </div>)}
        </div>}
      </div>}
    </Panel>
  );
}

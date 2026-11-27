import { useEffect, useState } from 'react';
import { BrainCircuit, Zap } from 'lucide-react';
import Panel from '../components/Panel.jsx';
import { api } from '../services/api.js';

export default function PredictionPage({ households = [] }) {
  const [meter, setMeter] = useState(households[0] || 'BR02');
  const [target, setTarget] = useState('next_24h');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (!meter && households.length) setMeter(households[0]); }, [households, meter]);

  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null);
    try { setResult(await api.predict({ meter, target })); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  return (
    <div className="page-stack narrow-page">
      <div className="page-heading"><div><p className="eyebrow">MODEL INFERENCE</p><h1>Electricity prediction</h1><p>Use the saved LightGBM / ensemble artifact for one selected household.</p></div></div>
      <Panel title="Generate a household forecast" subtitle="The local demo uses the latest saved feature row for the selected household.">
        <form className="form-grid" onSubmit={submit}>
          <label>Household meter<select value={meter} onChange={(e) => setMeter(e.target.value)}>{households.map((id) => <option key={id}>{id}</option>)}</select></label>
          <label>Prediction horizon<select value={target} onChange={(e) => setTarget(e.target.value)}><option value="next_24h">Next 24 hours</option><option value="next_30d">Next 30 days</option></select></label>
          <button className="button button-primary" disabled={busy || !meter}><BrainCircuit size={16} /> {busy ? 'Predicting…' : 'Generate prediction'}</button>
        </form>
      </Panel>
      {error && <div className="notice notice-error">{error}</div>}
      {result && <Panel title="Prediction result" subtitle="Model output is future interval energy in kWh.">
        <div className="prediction-result">
          <div><span>Predicted future electricity</span><strong>{Number(result.predicted_future_energy_kwh).toFixed(3)} <small>kWh</small></strong></div>
          <div><span>Household</span><strong>{result.meter}</strong></div>
          <div><span>Model</span><strong>{result.model_name}</strong></div>
          <div><span>Feature time</span><strong>{result.feature_timestamp_utc || result.feature_data_age_days ? result.feature_timestamp_utc || `${result.feature_data_age_days} days old` : 'Not provided'}</strong></div>
        </div>
        {result.warning && <div className="notice notice-warning">{result.warning}</div>}
        {result.stale_data_warning && <div className="notice notice-warning">This is a historical demonstration forecast. Upload more recent meter and weather records for a current forecast.</div>}
        {result.missing_input_fields?.length > 0 && <div className="notice notice-warning">Missing fields were imputed by the saved model: {result.missing_input_fields.join(', ')}</div>}
      </Panel>}
    </div>
  );
}

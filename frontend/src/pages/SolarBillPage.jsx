import { useState } from 'react';
import { Sun, Calculator } from 'lucide-react';
import Panel from '../components/Panel.jsx';
import { api } from '../services/api.js';

const categories = [
  ['LMV1_URBAN_DOMESTIC', 'Urban Domestic (LMV-1)'],
  ['LMV1_RURAL_DOMESTIC', 'Rural Domestic (LMV-1)'],
  ['LMV1_LIFE_LINE', 'Life Line (LMV-1)'],
  ['LMV2_RURAL_COMMERCIAL', 'Rural Commercial (LMV-2)'],
  ['LMV2_URBAN_COMMERCIAL_UP_TO_4KW', 'Urban Commercial ≤4 kW'],
  ['LMV2_URBAN_COMMERCIAL_ABOVE_4KW', 'Urban Commercial >4 kW'],
];

export default function SolarBillPage({ households = [] }) {
  const [form, setForm] = useState({ meter: households[0] || 'BR02', panel_capacity_kw: '3', peak_sun_hours: '4.5', connected_load_kw: '2', category_id: 'LMV1_URBAN_DOMESTIC', billing_days: '30', performance_ratio: '0.80' });
  const [result, setResult] = useState(null); const [error, setError] = useState(''); const [busy, setBusy] = useState(false);
  const set = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));
  async function submit(event) {
    event.preventDefault(); setBusy(true); setError(''); setResult(null);
    try { setResult(await api.solarBill({ ...form, panel_capacity_kw: Number(form.panel_capacity_kw), peak_sun_hours: Number(form.peak_sun_hours), connected_load_kw: Number(form.connected_load_kw), billing_days: Number(form.billing_days), performance_ratio: Number(form.performance_ratio) })); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }
  const solar = result?.solar; const bill = result?.bill;
  return (
    <div className="page-stack narrow-page">
      <div className="page-heading"><div><p className="eyebrow">SOLAR + TARIFF ESTIMATION</p><h1>Solar and bill estimator</h1><p>Combine the 30-day household model forecast with an irradiance-based solar estimate and the configured UPPCL tariff.</p></div></div>
      <Panel title="Household and rooftop configuration" subtitle="Peak sun hours should come from the selected location and intended forecast period. This form accepts that value as an input.">
        <form className="form-grid three-col" onSubmit={submit}>
          <label>Household meter<select value={form.meter} onChange={set('meter')}>{households.map((id) => <option key={id}>{id}</option>)}</select></label>
          <label>Panel capacity (kW)<input type="number" min="0.1" step="0.1" value={form.panel_capacity_kw} onChange={set('panel_capacity_kw')} required /></label>
          <label>Peak sun hours / day<input type="number" min="0" max="12" step="0.1" value={form.peak_sun_hours} onChange={set('peak_sun_hours')} required /></label>
          <label>Connected load (kW)<input type="number" min="0.1" step="0.1" value={form.connected_load_kw} onChange={set('connected_load_kw')} required /></label>
          <label>UPPCL tariff category<select value={form.category_id} onChange={set('category_id')}>{categories.map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
          <label>Billing period (days)<input type="number" min="28" max="31" value={form.billing_days} onChange={set('billing_days')} required /></label>
          <label>Performance ratio<input type="number" min="0.1" max="1" step="0.01" value={form.performance_ratio} onChange={set('performance_ratio')} required /></label>
          <button className="button button-primary" disabled={busy}><Calculator size={16} /> {busy ? 'Calculating…' : 'Estimate solar and bill'}</button>
        </form>
      </Panel>
      {error && <div className="notice notice-error">{error}</div>}
      {result && <>
        <div className="metric-grid three-metrics">
          <div className="result-tile"><span>Estimated solar generation</span><strong>{Number(solar.estimated_solar_generation_kwh).toFixed(2)} kWh</strong></div>
          <div className="result-tile"><span>Grid electricity required</span><strong>{Number(solar.estimated_grid_electricity_kwh).toFixed(2)} kWh</strong></div>
          <div className="result-tile result-primary"><span>Estimated UPPCL bill</span><strong>₹{Number(bill.estimated_total_bill_inr).toLocaleString('en-IN', { maximumFractionDigits: 2 })}</strong></div>
        </div>
        <Panel title="Bill breakdown" subtitle="Only configured energy slabs and fixed charges are included.">
          <div className="mini-metrics-table">
            <div className="table-head"><span>Component</span><span>Amount</span></div>
            <div className="table-row"><strong>Grid energy charge</strong><span>₹{bill.energy_charge_inr}</span></div>
            <div className="table-row"><strong>Fixed charge</strong><span>₹{bill.fixed_charge_inr}</span></div>
            <div className="table-row"><strong>Other configured charges</strong><span>₹{bill.other_charges_inr}</span></div>
          </div>
          {(result.electricity_forecast.warning || result.electricity_forecast.stale_data_warning) && <div className="notice notice-warning">The electricity features come from historical data. Upload new meter and weather data for a current forecast.</div>}
          {bill.warnings?.map((warning) => <div className="notice notice-warning" key={warning}>{warning}</div>)}
          <div className="notice notice-info"><Sun size={16} /> {result.weather_assumption} This is a period-level energy estimate and does not model intraday load timing, battery storage or export credits.</div>
        </Panel>
      </>}
    </div>
  );
}

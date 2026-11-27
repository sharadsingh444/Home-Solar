import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';

export default function MetricCard({ label, value, suffix = '', detail, tone = 'blue', icon: Icon, change }) {
  const Trend = change > 0 ? ArrowUpRight : change < 0 ? ArrowDownRight : Minus;
  return (
    <section className={`metric-card tone-${tone}`}>
      <div className="metric-top">
        <span className="metric-label">{label}</span>
        {Icon && <span className="metric-icon"><Icon size={18} /></span>}
      </div>
      <div className="metric-value">{value ?? '—'}<small>{suffix}</small></div>
      <div className="metric-foot">
        <span>{detail || 'No result recorded yet'}</span>
        {Number.isFinite(change) && <span className={change <= 0 ? 'trend-good' : 'trend-bad'}><Trend size={14} /> {Math.abs(change).toFixed(1)}%</span>}
      </div>
    </section>
  );
}

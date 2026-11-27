import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

export default function ImprovementChart({ history = [] }) {
  const rows = [];
  for (const run of history) {
    for (const [target, outcome] of Object.entries(run.targets || {})) {
      const comparison = outcome.comparison;
      if (!comparison?.baseline || !comparison?.candidate) continue;
      rows.push({
        label: `${String(run.run_id || '').slice(0, 13)} ${target === 'next_24h' ? '24h' : '30d'}`,
        baseline: Number(comparison.baseline.mae_kwh),
        candidate: Number(comparison.candidate.mae_kwh),
        promoted: outcome.status === 'promoted',
      });
    }
  }
  if (!rows.length) return <div className="empty-state">No retraining comparison has been recorded. Upload a new electricity batch and its matching weather batch to start the self-development cycle.</div>;
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={rows} margin={{ top: 12, right: 20, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e7edf3" />
          <XAxis dataKey="label" tick={{ fill: '#66788a', fontSize: 11 }} />
          <YAxis tick={{ fill: '#66788a', fontSize: 12 }} />
          <Tooltip formatter={(v) => `${Number(v).toFixed(3)} kWh MAE`} />
          <Legend />
          <Line type="monotone" dataKey="baseline" name="Existing model MAE" stroke="#d97757" strokeWidth={2.5} dot={{ r: 4 }} />
          <Line type="monotone" dataKey="candidate" name="New candidate MAE" stroke="#159879" strokeWidth={2.5} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
      <p className="chart-note">Each pair is evaluated on the same newly arrived held-out rows. A lower candidate line means the candidate made smaller errors.</p>
    </div>
  );
}

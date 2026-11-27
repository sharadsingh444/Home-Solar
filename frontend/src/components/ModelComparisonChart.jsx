import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const fmt = (value) => `${Number(value).toFixed(2)} kWh`;

export default function ModelComparisonChart({ models = {} }) {
  const data = ['next_24h', 'next_30d'].map((key) => {
    const item = models[key] || {};
    const test = item.test_metrics || {};
    return {
      horizon: key === 'next_24h' ? 'Next 24 hours' : 'Next 30 days',
      MAE: Number.isFinite(Number(test.mae_kwh)) ? Number(test.mae_kwh) : null,
      RMSE: Number.isFinite(Number(test.rmse_kwh)) ? Number(test.rmse_kwh) : null,
    };
  }).filter((row) => row.MAE !== null || row.RMSE !== null);

  if (!data.length) return <div className="empty-state">No held-out test metrics are available yet. Run ml/evaluate.py for each horizon.</div>;
  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={275}>
        <BarChart data={data} margin={{ top: 12, right: 12, left: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e7edf3" />
          <XAxis dataKey="horizon" tick={{ fill: '#66788a', fontSize: 12 }} />
          <YAxis tick={{ fill: '#66788a', fontSize: 12 }} />
          <Tooltip formatter={fmt} />
          <Legend />
          <Bar dataKey="MAE" fill="#1f9d83" radius={[6, 6, 0, 0]} />
          <Bar dataKey="RMSE" fill="#6793db" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="chart-note">Current saved evaluation files. Lower values indicate smaller prediction errors.</p>
    </div>
  );
}

const API_BASE = import.meta.env.VITE_API_BASE || '';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  let body;
  try { body = text ? JSON.parse(text) : {}; }
  catch { body = { error: text || 'Invalid API response' }; }
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

export const api = {
  health: () => request('/api/health'),
  dashboard: () => request('/api/dashboard/summary'),
  history: () => request('/api/dashboard/history'),
  predict: (payload) => request('/api/predictions/energy', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }),
  solarBill: (payload) => request('/api/predictions/solar-bill', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }),
  startUpdate: (electricityFile, weatherFile) => {
    const form = new FormData();
    form.append('electricityCsv', electricityFile);
    form.append('weatherCsv', weatherFile);
    return request('/api/updates/run', { method: 'POST', body: form });
  },
  updateJob: (id) => request(`/api/updates/jobs/${id}`),
};

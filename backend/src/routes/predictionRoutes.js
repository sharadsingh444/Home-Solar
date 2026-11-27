import { Router } from 'express';
import { runPythonWorker } from '../services/pythonBridge.js';

const router = Router();

router.post('/energy', async (req, res, next) => {
  try {
    const { meter, target = 'next_24h' } = req.body || {};
    if (!meter || !String(meter).trim()) return res.status(400).json({ error: 'meter is required.' });
    if (!['next_24h', 'next_30d'].includes(target)) return res.status(400).json({ error: 'target must be next_24h or next_30d.' });
    const result = await runPythonWorker('predict', { meter, target });
    res.json(result);
  } catch (error) { next(error); }
});

router.post('/solar-bill', async (req, res, next) => {
  try {
    const { meter, panel_capacity_kw, peak_sun_hours, connected_load_kw, category_id, billing_days = 30, performance_ratio = 0.80 } = req.body || {};
    if (!meter || !category_id) return res.status(400).json({ error: 'meter and category_id are required.' });
    for (const [name, value] of Object.entries({ panel_capacity_kw, peak_sun_hours, connected_load_kw, billing_days, performance_ratio })) {
      if (value === undefined || !Number.isFinite(Number(value))) return res.status(400).json({ error: `${name} must be numeric.` });
    }
    const result = await runPythonWorker('solar-bill', {
      meter,
      panel_capacity_kw: Number(panel_capacity_kw),
      peak_sun_hours: Number(peak_sun_hours),
      connected_load_kw: Number(connected_load_kw),
      category_id,
      billing_days: Number(billing_days),
      performance_ratio: Number(performance_ratio),
    });
    res.json(result);
  } catch (error) { next(error); }
});

export default router;

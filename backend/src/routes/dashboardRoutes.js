import { Router } from 'express';
import { runPythonWorker } from '../services/pythonBridge.js';

const router = Router();

router.get('/summary', async (req, res, next) => {
  try {
    const summary = await runPythonWorker('dashboard', {}, { timeoutMs: 180_000 });
    res.json(summary);
  } catch (error) { next(error); }
});

router.get('/history', async (req, res, next) => {
  try {
    const history = await runPythonWorker('history');
    res.json({ history });
  } catch (error) { next(error); }
});

export default router;

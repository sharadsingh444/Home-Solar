import { Router } from 'express';
import multer from 'multer';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { incomingElectricityDir, incomingWeatherDir } from '../config/paths.js';
import { runPythonWorker } from '../services/pythonBridge.js';
import { createUpdateJob, getUpdateJob, updateJob } from '../state/updateJobs.js';

const router = Router();
fs.mkdirSync(incomingElectricityDir, { recursive: true });
fs.mkdirSync(incomingWeatherDir, { recursive: true });

const storage = multer.diskStorage({
  destination: (req, file, callback) => callback(null, file.fieldname === 'electricityCsv' ? incomingElectricityDir : incomingWeatherDir),
  filename: (req, file, callback) => callback(null, `${new Date().toISOString().replace(/[:.]/g, '-')}_${randomUUID()}_${path.basename(file.originalname).replace(/[^a-zA-Z0-9._-]/g, '_')}`),
});
const upload = multer({
  storage,
  limits: { fileSize: 1024 * 1024 * 1024, files: 2 },
  fileFilter: (req, file, callback) => {
    if (path.extname(file.originalname).toLowerCase() !== '.csv') { const error = new Error('Only .csv files are accepted.'); error.status = 400; return callback(error); }
    callback(null, true);
  },
});

router.post('/run', upload.fields([
  { name: 'electricityCsv', maxCount: 1 },
  { name: 'weatherCsv', maxCount: 1 },
]), (req, res, next) => {
  try {
    const electricityFile = req.files?.electricityCsv?.[0];
    const weatherFile = req.files?.weatherCsv?.[0];
    if (!electricityFile || !weatherFile) {
      return res.status(400).json({ error: 'Upload both electricityCsv and weatherCsv. The weather batch is required so new meter readings can be matched to weather.' });
    }
    const job = createUpdateJob();
    updateJob(job.id, { status: 'running', stage: 'ETL → feature generation → candidate training → comparison' });
    res.status(202).json({
      ...getUpdateJob(job.id),
      archivedFiles: {
        electricity: path.relative(process.cwd(), electricityFile.path),
        weather: path.relative(process.cwd(), weatherFile.path),
      },
      message: 'Update started. Poll GET /api/updates/jobs/:id for the result.',
    });

    runPythonWorker('update', {
      new_electricity: electricityFile.path,
      new_weather: weatherFile.path,
      min_improvement_pct: Number(process.env.HOMESOLAR_MIN_MODEL_IMPROVEMENT_PCT || 0.5),
    }, { timeoutMs: 60 * 60 * 1000 })
      .then((result) => updateJob(job.id, { status: 'completed', stage: 'Update workflow finished', result }))
      .catch((error) => updateJob(job.id, { status: 'failed', stage: 'Update workflow stopped', error: error.message }));
  } catch (error) { next(error); }
});

router.get('/jobs/:id', (req, res) => {
  const job = getUpdateJob(req.params.id);
  if (!job) return res.status(404).json({ error: 'Update job not found. Jobs are held in server memory until restart.' });
  res.json(job);
});

export default router;

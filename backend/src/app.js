import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dashboardRoutes from './routes/dashboardRoutes.js';
import predictionRoutes from './routes/predictionRoutes.js';
import updateRoutes from './routes/updateRoutes.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandler.js';

const backendDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const frontendDist = path.resolve(backendDir, '..', 'frontend', 'dist');

const app = express();
const allowedOrigin = process.env.CORS_ORIGIN || 'http://localhost:5173';

app.use(helmet({ crossOriginResourcePolicy: { policy: 'cross-origin' } }));
app.use(cors({ origin: allowedOrigin.split(',').map((item) => item.trim()), credentials: false }));
app.use(express.json({ limit: '12mb' }));
app.use(morgan('dev'));

app.get('/api/health', (req, res) => res.json({ ok: true, service: 'HomeSolar Express API', timestamp: new Date().toISOString() }));
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/predictions', predictionRoutes);
app.use('/api/updates', updateRoutes);

// In production, Express can serve the built React application from frontend/dist.
if (process.env.NODE_ENV === 'production' && fs.existsSync(frontendDist)) {
  app.use(express.static(frontendDist));
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api/')) return next();
    res.sendFile(path.join(frontendDist, 'index.html'));
  });
}

app.use(notFoundHandler);
app.use(errorHandler);

export default app;

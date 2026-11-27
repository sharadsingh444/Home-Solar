import 'dotenv/config';
import app from './src/app.js';

const port = Number(process.env.PORT || 5000);
const server = app.listen(port, () => {
  console.log(`HomeSolar Express API listening at http://localhost:${port}`);
  console.log('Python ML and ETL jobs run through backend/python_worker.py');
});
// Model updates may take longer than the default HTTP timeout. The update
// endpoint returns 202 immediately and exposes a polling job resource.
server.requestTimeout = 0;
server.headersTimeout = 65_000;

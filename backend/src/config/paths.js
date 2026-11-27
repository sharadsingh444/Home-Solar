import 'dotenv/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentFile = fileURLToPath(import.meta.url);
const backendDir = path.resolve(path.dirname(currentFile), '../../');
export const projectRoot = path.resolve(backendDir, '..');
export const backendRoot = backendDir;
export const pythonWorkerPath = path.join(backendDir, 'python_worker.py');
export const incomingElectricityDir = path.join(projectRoot, 'data', 'raw', 'incoming', 'electricity');
export const incomingWeatherDir = path.join(projectRoot, 'data', 'raw', 'incoming', 'weather');

const configuredPython = process.env.HOMESOLAR_PYTHON || process.env.PYTHON_EXECUTABLE;
const windowsPython = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
const posixPython = path.join(projectRoot, '.venv', 'bin', 'python');
export const pythonExecutable = configuredPython || (process.platform === 'win32' ? windowsPython : posixPython);

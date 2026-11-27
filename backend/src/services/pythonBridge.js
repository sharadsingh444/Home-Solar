import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { projectRoot, pythonExecutable, pythonWorkerPath } from '../config/paths.js';

function resolvePython() {
  if (pythonExecutable && fs.existsSync(pythonExecutable)) return pythonExecutable;
  // PATH fallback is useful when the virtual environment has not been created at the project root.
  return process.platform === 'win32' ? 'python' : 'python3';
}

export function runPythonWorker(command, payload = {}, { timeoutMs = 120_000 } = {}) {
  return new Promise((resolve, reject) => {
    const executable = resolvePython();
    const child = spawn(executable, [pythonWorkerPath, command], {
      cwd: projectRoot,
      env: process.env,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const timeout = setTimeout(() => {
      if (!settled) {
        child.kill('SIGTERM');
        reject(new Error(`Python worker '${command}' exceeded ${Math.round(timeoutMs / 1000)} seconds.`));
        settled = true;
      }
    }, timeoutMs);

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (part) => {
      stdout += part;
      if (stdout.length > 15_000_000) {
        child.kill('SIGTERM');
        if (!settled) reject(new Error('Python worker returned too much output.'));
        settled = true;
      }
    });
    child.stderr.on('data', (part) => {
      stderr += part;
      if (stderr.length > 2_000_000) stderr = stderr.slice(-2_000_000);
    });
    child.on('error', (error) => {
      clearTimeout(timeout);
      if (!settled) reject(new Error(`Could not start Python (${executable}): ${error.message}`));
      settled = true;
    });
    child.on('close', (code) => {
      clearTimeout(timeout);
      if (settled) return;
      settled = true;
      let response;
      try {
        response = JSON.parse(stdout.trim().split('\n').filter(Boolean).at(-1) || '{}');
      } catch {
        reject(new Error(`Python worker returned invalid JSON. ${stderr.slice(-5000)} ${stdout.slice(-1000)}`));
        return;
      }
      if (code !== 0 || response.ok === false) {
        reject(new Error(response.error || `Python worker exited with code ${code}. ${stderr.slice(-6000)}`));
        return;
      }
      resolve(response.result);
    });

    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

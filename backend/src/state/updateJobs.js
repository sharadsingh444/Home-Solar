import { randomUUID } from 'node:crypto';

const jobs = new Map();

export function createUpdateJob() {
  const id = randomUUID();
  const job = {
    id,
    status: 'queued',
    stage: 'Files received; waiting for update worker',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    result: null,
    error: null,
  };
  jobs.set(id, job);
  return job;
}

export function updateJob(id, patch) {
  const previous = jobs.get(id);
  if (!previous) return null;
  const next = { ...previous, ...patch, updatedAt: new Date().toISOString() };
  jobs.set(id, next);
  return next;
}

export function getUpdateJob(id) {
  return jobs.get(id) || null;
}

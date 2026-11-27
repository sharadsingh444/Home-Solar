export function notFoundHandler(req, res) {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.originalUrl}` });
}

export function errorHandler(error, req, res, next) {
  if (res.headersSent) return next(error);
  const inferred = error.code === 'LIMIT_FILE_SIZE' ? 413 : (error instanceof SyntaxError ? 400 : 500);
  const status = Number(error.status || error.statusCode || inferred);
  const message = error.message || 'Unexpected server error';
  if (status >= 500) console.error(error);
  res.status(status).json({ error: message });
}

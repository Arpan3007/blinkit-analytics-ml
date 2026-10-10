export const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api';

async function _post(path: string, body: any) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── CSV ──────────────────────────────────────────────────────────────────────
export async function uploadCSV(file: File) {
  const fd = new FormData(); fd.append('file', file);
  const r = await fetch(`${API}/datasets/upload`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

// ── ML ───────────────────────────────────────────────────────────────────────
export async function train(payload: any)         { return _post('/ml/train', payload); }
export async function predict(payload: any)       { return _post('/ml/predict', payload); }
export async function predictCustom(payload: any) { return _post('/ml/predict_custom', payload); }
export async function getAIInsights(payload: any) { return _post('/ml/ai_insights', payload); }

// ── KAGGLE ───────────────────────────────────────────────────────────────────
export async function kaggleListFiles(url: string) {
  return _post('/kaggle/list-files', { url });
}
export async function kaggleImportFile(download_id: string, filename: string) {
  return _post('/kaggle/import-file', { download_id, filename });
}
export async function kaggleCleanup(download_id: string) {
  const r = await fetch(`${API}/kaggle/cleanup/${download_id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}


// ── PREPROCESS ───────────────────────────────────────────────────────────────
export async function preprocessDataset(dataset_id: string, operations: string[]) {
  return _post('/datasets/preprocess', { dataset_id, operations });
}

// ── EXPORT ───────────────────────────────────────────────────────────────────
export function exportDatasetUrl(dataset_id: string, filename: string) {
  return `${API}/datasets/${dataset_id}/export?filename=${encodeURIComponent(filename)}`;
}

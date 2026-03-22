const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function analyzeWithBackend(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  const data = await res.json();
  if (!res.ok || data?.error) {
    throw new Error(data?.error || `Backend error (${res.status})`);
  }
  return data; // { timeline: [...], sampleRate: ... }
}
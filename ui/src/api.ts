import type { AppConfig, CouplingRun, FileRow, ModelGroup, RunRow, Session, SolverCheck } from './types'

async function j<T>(r: Response): Promise<T> {
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error((body as { detail?: string }).detail || `${r.status} ${r.statusText}`)
  return body as T
}

export const api = {
  config: () => fetch('/api/config').then((r) => j<AppConfig>(r)),
  models: () => fetch('/api/models').then((r) => j<{ groups: ModelGroup[]; default: string | null }>(r)),
  solvers: (refresh = false) =>
    fetch(`/api/solvers${refresh ? '?refresh=true' : ''}`)
      .then((r) => j<SolverCheck>(r)),
  runs: () => fetch('/api/sessions').then((r) => j<{ sessions: RunRow[]; running: number }>(r)),
  createRun: (model: string, mode: string) =>
    fetch('/api/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                             body: JSON.stringify({ model, mode }) }).then((r) => j<Session>(r)),
  deleteRun: (id: string) => fetch(`/api/sessions/${id}`, { method: 'DELETE' }).then((r) => j(r)),
  // the first message goes to the server before the page moves, so closing the
  // tab in between cannot lose it
  startRun: (id: string, text: string, attachments: string[]) =>
    fetch(`/api/sessions/${id}/prompt`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                          body: JSON.stringify({ text, attachments }) }).then((r) => j(r)),
  files: (id: string, sub = '') =>
    fetch(`/api/sessions/${id}/files?sub=${encodeURIComponent(sub)}`)
      .then((r) => j<{ entries: FileRow[]; sub: string; exists: boolean }>(r)),
  upload: (id: string, files: File[]) => {
    const fd = new FormData()
    files.forEach((f) => fd.append('files', f, f.name))
    return fetch(`/api/sessions/${id}/upload`, { method: 'POST', body: fd })
      .then((r) => j<{ saved: { name: string; bytes: number }[] }>(r))
  },
  viz: (rel: string) => fetch(`/api/viz?rel=${encodeURIComponent(rel)}`).then((r) => j<{ kind: string } & Record<string, unknown>>(r)),
  coupling: (id: string) => fetch(`/api/sessions/${id}/coupling`).then((r) => j<{ runs: CouplingRun[] }>(r)),
  manifestUrl: (id: string) => `/api/sessions/${id}/manifest`,
}

export function socket(id: string): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  return new WebSocket(`${proto}://${location.host}/ws/${id}`)
}

/* The home directory is shown as ~. A transcript is something people screenshot
   and share, and whose machine it ran on is nobody else's business. */
/** Home directories out of anything shown. Text is otherwise left as written:
    turning a literal backslash-n into a line break broke LaTeX such as \nabla. */
/** A run's own file, as a URL. A name may contain ? or #, which the browser
    would read as a query or fragment and ask for a shorter path than meant. */
export function fileUrl(rel: string): string {
  return '/sandbox-file/' + rel.split('/').map(encodeURIComponent).join('/')
}

export function tidy(s: string): string {
  return s
    // a path inside the run's own folder is shown from that folder: the absolute prefix says
    // nothing a reader needs and names the machine's layout
    .replace(/(?:[A-Za-z]:)?[^\s'"`=(]*?eval_interactive[\\/]+webui_[0-9a-f]{6,}(?:[\\/]+|(?=[\s'"`),:]|$))/g, '')
    .replace(/(?:\b[A-Za-z]:|\/[a-z](?=\/))[\\/]+Users[\\/]+(?:[^\s\\/:*?"<>|'`]+(?: [^\s\\/:*?"<>|'`]+)*(?=[\\/])|[^\s\\/:*?"<>|'`]+)/gi, '~')
    .replace(/\/(?:home|Users)\/[^/\s:'"]+/g, '~')
}

export function ago(ts: number): string {
  const s = Date.now() / 1000 - ts
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

export function clock(seconds: number): string {
  const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60)
  return m ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`
}

export function money(usd?: number | null): string | null {
  if (usd == null) return null
  return usd < 0.01 ? '< $0.01' : `$${usd.toFixed(2)}`
}

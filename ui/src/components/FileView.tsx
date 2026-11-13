import { useEffect, useMemo, useRef, useState } from 'react'
import { fileUrl } from '../api'
import { open3d } from '../open3d'

/* Whatever the run wrote, readable.

   Clicking a file used to be a silent no-op unless the payload happened to be a
   field animation, so the script the agent wrote, the mesh, the log and the
   tables were listed and unopenable. The bytes were always served; nothing
   asked for them. */

type Viz =
  | { kind: 'text'; text: string; syntax?: string; truncated?: boolean }
  | { kind: 'json'; obj: unknown }
  | { kind: 'table'; rows: string[][]; header: string[]; truncated?: boolean }
  | { kind: 'image'; rel: string }
  | { kind: 'error'; error: string }
  | { kind: string; [k: string]: unknown }

/** A plain line chart. No library, no chart chrome, one series per column. */
function Chart({ header, rows }: { header: string[]; rows: string[][] }) {
  const series = useMemo(() => {
    const xs: number[] = []
    const cols: number[][] = header.slice(1).map(() => [])
    for (const r of rows) {
      const x = Number(r[0])
      if (!Number.isFinite(x)) continue
      xs.push(x)
      // Number('') is 0, so a blank cell used to be drawn as a real zero: the
      // chart invented data a solver never wrote
      header.slice(1).forEach((_, i) => {
        const cell = (r[i + 1] ?? '').trim()
        cols[i].push(cell === '' ? NaN : Number(cell))
      })
    }
    return { xs, cols }
  }, [header, rows])

  // a line only means something along an ordered axis; a table of x, y, value
  // points drawn as a line is a meaningless zig-zag
  const ordered = series.xs.every((x, i) => i === 0 || x >= series.xs[i - 1])
  if (series.xs.length < 2 || !ordered || new Set(series.xs).size < series.xs.length * 0.9) return null
  // a table may hold words or gaps; only columns that are really numbers are
  // drawn, and a table with none is shown as a table and nothing else
  const drawn = series.cols
    .map((col, i) => ({ name: header[i + 1], col }))
    .filter(({ col }) => col.filter(Number.isFinite).length >= 2)
  const all = drawn.flatMap(({ col }) => col).filter(Number.isFinite)
  if (!drawn.length || !all.length) return null
  const W = 640, H = 240, P = 32
  const xmin = Math.min(...series.xs), xmax = Math.max(...series.xs)
  const ymin = Math.min(...all), ymax = Math.max(...all)
  const sx = (v: number) => P + (v - xmin) / (xmax - xmin || 1) * (W - 2 * P)
  const sy = (v: number) => H - P - (v - ymin) / (ymax - ymin || 1) * (H - 2 * P)
  const stroke = ['#FF6B4A', '#94A3B8', '#AFBCCB']

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full mt-4" role="img"
         aria-label={`${drawn.map((d) => d.name).join(', ')} against ${header[0]}`}>
      <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="rgb(100 116 139 / .28)" />
      <line x1={P} y1={P} x2={P} y2={H - P} stroke="rgb(100 116 139 / .28)" />
      {drawn.map(({ col }, i) => {
        // one line per run of real values: dropping the gaps and joining what
        // is left draws a straight line across missing data, which reads as a
        // measurement between two points that the solver never wrote
        const runs: string[][] = []
        col.forEach((v, j) => {
          if (!Number.isFinite(v)) { runs.push([]); return }
          if (!runs.length) runs.push([])
          runs[runs.length - 1].push(`${sx(series.xs[j])},${sy(v)}`)
        })
        return runs.filter((r) => r.length > 1).map((points, k) => (
          <polyline key={`${i}-${k}`} fill="none" strokeWidth="1.5"
                    stroke={stroke[i % stroke.length]} points={points.join(' ')} />
        ))
      })}
      <text x={P} y={H - 10} fill="#94A3B8" fontSize="11" fontFamily="monospace">
        {header[0]}
      </text>
      <text x={P} y={P - 10} fill="#94A3B8" fontSize="11" fontFamily="monospace">
        {ymax.toPrecision(4)}
      </text>
    </svg>
  )
}

export default function FileView({ rel, onClose }: { rel: string; onClose: () => void }) {
  const [viz, setViz] = useState<Viz | null>(null)
  const panel = useRef<HTMLElement>(null)

  // it announces itself as a modal, so it has to behave as one: focus moves in,
  // Tab stays inside, and whatever opened it gets focus back
  // the run view hands down a new onClose every second while a run works
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const returnTo = document.activeElement as HTMLElement | null
    panel.current?.focus()
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); close.current(); return }
      if (e.key !== 'Tab') return
      const inside = panel.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])')
      if (!inside?.length) return
      const first = inside[0], last = inside[inside.length - 1]
      const here = document.activeElement
      if (!e.shiftKey && (here === last || here === panel.current)) { e.preventDefault(); first.focus() }
      else if (e.shiftKey && (here === first || here === panel.current)) { e.preventDefault(); last.focus() }
    }
    document.addEventListener('keydown', key, true)
    return () => { document.removeEventListener('keydown', key, true); returnTo?.focus?.() }
  }, [rel])

  useEffect(() => {
    // opening files quickly could show the first file's contents under the
    // second file's name, whichever answer happened to arrive last
    setViz(null)
    const stop = new AbortController()
    fetch(`/api/viz?rel=${encodeURIComponent(rel)}`, { signal: stop.signal })
      .then((r) => r.json())
      .then(setViz)
      .catch((e) => { if (!stop.signal.aborted) setViz({ kind: 'error', error: String(e) }) })
    return () => stop.abort()
  }, [rel])

  const name = rel.split('/').pop() || rel

  return (
    <div className="fixed inset-0 z-30 flex" role="dialog" aria-modal="true" aria-label={name}>
      <div className="flex-1 bg-black/55" onClick={onClose} />
      <aside ref={panel} tabIndex={-1}
             className="w-[760px] h-full bg-card border-l line overflow-y-auto scroll p-8
                        overscroll-contain outline-none">
        <div className="flex items-center gap-4">
          <span className="num text-[14px] text-ink2 min-w-0 overflow-hidden
                           text-ellipsis whitespace-nowrap">{name}</span>
          <a href={fileUrl(rel)} download
             className="ml-auto h-8 px-3.5 rounded-[6px] border line text-[13px] text-muted
                        grid place-items-center transition-colors duration-150
                        hover:text-ink2">
            Download
          </a>
          <button onClick={onClose}
                  className="h-8 px-3.5 rounded-[6px] border line text-[13px] text-muted
                             transition-colors duration-150 hover:text-ink2">Close</button>
        </div>
        <div className="num text-[13px] text-muted mt-2">{rel}</div>

        <div className="mt-6">
          {!viz && <div className="text-[14px] text-muted">Opening…</div>}

          {viz?.kind === 'error' && (
            <div className="text-[14px] text-[#D85A6F]">{String(viz.error)}</div>
          )}

          {viz?.kind === 'text' && (viz as { truncated?: boolean }).truncated && (
            <p className="text-[14px] text-coral mb-3">
              The beginning of a longer file. Download it to see all of it.
            </p>
          )}

          {viz?.kind === 'text' && (
            <pre className="num text-[13px] leading-[1.65] text-body whitespace-pre-wrap
                            break-words">{(viz as { text: string }).text}</pre>
          )}

          {viz?.kind === 'json' && (
            <pre className="num text-[13px] leading-[1.65] text-body whitespace-pre-wrap">
              {JSON.stringify((viz as { obj: unknown }).obj, null, 2)}
            </pre>
          )}

          {viz?.kind === 'table' && (
            <>
              {(viz as { truncated?: boolean }).truncated && (
                <p className="text-[14px] text-coral mb-3">
                  The first {(viz as { rows: string[][] }).rows.length.toLocaleString()} rows of a longer
                  file. Download it for the rest — what is drawn and shown below stops here.
                </p>
              )}
              <Chart header={(viz as { header: string[] }).header}
                     rows={(viz as { rows: string[][] }).rows} />
              <table className="w-full mt-6 num text-[13px]">
                <thead>
                  <tr>{(viz as { header: string[] }).header.map((h) => (
                    <th key={h} className="text-left text-muted font-normal pb-2
                                           border-b line-soft">{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {(viz as { rows: string[][] }).rows.slice(0, 40).map((r, i) => (
                    <tr key={i}>{r.map((c, j) => (
                      <td key={j} className="text-body py-1 border-b line-soft">{c}</td>
                    ))}</tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {viz?.kind === 'image' && (
            <img src={fileUrl(rel)} alt={name}
                 className="max-w-full rounded-[6px] border line" />
          )}

          {viz?.kind === 'vtk' && (
            <div className="text-[15px] leading-[1.55] text-body">
              <p>A result file in {String(viz.format || 'VTK').toUpperCase()} format: a mesh and the values the solver stored on it.</p>
              <button onClick={() => { open3d(rel); onClose() }}
                      className="mt-3 h-9 px-4 rounded-[8px] bg-coral text-on-coral text-[14px] font-semibold hover:bg-coral-h">Open 3D view</button>
              <p className="mt-3 text-muted">The 3D view draws the outer surface in your browser. For cuts through the inside, download it and open it in ParaView, VisIt or PyVista.</p>
            </div>
          )}

          {viz?.kind === 'hdf' && (
            <div className="text-[15px] leading-[1.55] text-body">
              <p>A data file in HDF5 format. openPASO does not draw it here.</p>
              {Array.isArray(viz.keys) && viz.keys.length > 0 && (
                <p className="num mt-2 text-[14px] text-muted break-all">It contains: {(viz.keys as string[]).join(', ')}</p>
              )}
              <p className="mt-2 text-muted">Download it and open it in ParaView or h5py.</p>
            </div>
          )}

          {viz?.kind === 'xdmf' && (
            <div className="text-[15px] leading-[1.55] text-body">
              <p>An XDMF file: it describes a mesh and fields, and the numbers live in the data files beside it.</p>
              {Array.isArray(viz.data_files) && viz.data_files.length > 0 && (
                <p className="num mt-2 text-[14px] text-muted break-all">It points at: {(viz.data_files as string[]).join(', ')}</p>
              )}
              <pre className="num text-[13px] leading-[1.6] text-body mt-3 p-4 bg-soft border line rounded-[8px]
                              max-h-[420px] overflow-auto scroll whitespace-pre-wrap break-words">{String(viz.text || '')}</pre>
            </div>
          )}

          {viz && !['text', 'json', 'table', 'image', 'error', 'vtk', 'hdf', 'xdmf'].includes(viz.kind) && (
            <div className="text-[15px] text-body">
              openPASO cannot show a {viz.kind} file here. Download it and open it in a program that reads it.
            </div>
          )}

          {viz && !['text', 'json', 'table', 'error', 'xdmf'].includes(viz.kind) && (
            <p className="mt-6 pt-4 border-t line text-[14px] text-muted">
              Text files are stripped of home directories on their way to you. This one is sent exactly as the
              run wrote it, so a path can still be inside it.
            </p>
          )}
        </div>
      </aside>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { api } from '../api'
import type { CouplingRun } from '../types'

/* The coupling's residual, iteration by iteration, while it iterates.

   A coupled solve can run for an hour with nothing on the page but a clock. The driver
   appends every iteration's residual to a file beside the participants; this reads it
   every two seconds while the step runs, and once after. Each call of the driver (one
   per mesh level under couple_levels) is one line. The first iteration has no residual
   (there is nothing to compare with yet) and is not drawn. */

const W = 640, H = 220, L = 52, R = 16, T = 14, B = 34

export default function CouplingPlot({ runId, t0, t1, live }: {
  runId: string; t0?: number; t1?: number; live: boolean
}) {
  const [runs, setRuns] = useState<CouplingRun[]>([])
  useEffect(() => {
    let dead = false
    const load = () => api.coupling(runId)
      .then((d) => { if (!dead) setRuns(d.runs) })
      .catch(() => { /* nothing to show yet */ })
    load()
    if (!live) return () => { dead = true }
    const id = window.setInterval(load, 2000)
    return () => { dead = true; window.clearInterval(id) }
  }, [runId, live])

  // only the couplings this step started: event times are in ms, the driver's in seconds
  const mine = runs.filter((r) => {
    const s = r.points[0][2] * 1000
    return (t0 == null || s >= t0 - 2000) && (t1 == null || s <= t1 + 2000)
  })
  const pts = mine.map((r) => r.points.filter((p): p is [number, number, number] => p[1] != null && p[1] > 0))
  const all = pts.flat()
  if (!mine.length) return null
  const tol = mine[mine.length - 1].tol
  const last = mine[mine.length - 1].points[mine[mine.length - 1].points.length - 1]

  const itMax = Math.max(2, ...mine.map((r) => r.points[r.points.length - 1][0]))
  const logs = all.map((p) => Math.log10(p[1]))
  if (tol && tol > 0) logs.push(Math.log10(tol))
  const lo = logs.length ? Math.floor(Math.min(...logs)) : -6
  const hi = logs.length ? Math.max(lo + 1, Math.ceil(Math.max(...logs))) : 0
  const x = (it: number) => L + ((it - 1) / (itMax - 1)) * (W - L - R)
  const y = (v: number) => T + ((hi - Math.log10(v)) / (hi - lo)) * (H - T - B)
  const decades = Array.from({ length: hi - lo + 1 }, (_, k) => hi - k)
  const step = Math.max(1, Math.ceil((itMax - 1) / 8))
  const iters = Array.from({ length: Math.floor((itMax - 1) / step) + 1 }, (_, k) => 1 + k * step)

  return (
    <figure className="pl-7 mt-3" aria-label="Coupling residual per iteration">
      <div className="bg-soft border line rounded-[8px] p-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto block" role="img"
             aria-label={`Residual of ${mine.length} coupling${mine.length > 1 ? 's' : ''}, logarithmic`}>
          <g className="text-hairline" stroke="currentColor" strokeWidth={1}>
            {decades.map((d) => <line key={d} x1={L} x2={W - R} y1={y(10 ** d)} y2={y(10 ** d)} />)}
          </g>
          <g className="text-muted num" fill="currentColor" fontSize={12}>
            {decades.map((d) => <text key={d} x={L - 8} y={y(10 ** d) + 4} textAnchor="end">1e{d}</text>)}
            {iters.map((i) => <text key={i} x={x(i)} y={H - B + 18} textAnchor="middle">{i}</text>)}
            <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle">iteration</text>
          </g>
          {tol != null && tol > 0 && (
            <g className="text-coral" stroke="currentColor" fill="currentColor">
              <line x1={L} x2={W - R} y1={y(tol)} y2={y(tol)} strokeDasharray="5 4" strokeWidth={1.2} />
              <text x={L + 6} y={y(tol) - 5} textAnchor="start" fontSize={12} stroke="none">target {tol.toExponential(0)}</text>
            </g>
          )}
          {pts.map((p, k) => (
            <g key={mine[k].run} className={k === pts.length - 1 ? 'text-ink2' : 'text-muted'}
               stroke="currentColor" fill="currentColor">
              <polyline fill="none" strokeWidth={k === pts.length - 1 ? 2 : 1.4}
                        points={p.map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')} />
              {p.map(([i, v]) => <circle key={i} cx={x(i)} cy={y(v)} r={2.4} stroke="none" />)}
            </g>
          ))}
        </svg>
      </div>
      <figcaption className="num mt-1.5 text-[13px] text-muted">
        {mine.length > 1 ? `${mine.length} couplings (one per call; the latest in dark) · ` : ''}
        {live ? 'now at ' : 'ended at '}iteration {last[0]}
        {last[1] != null ? ` · residual ${last[1].toExponential(2)}` : ''}
        {tol != null ? ` · target ${tol.toExponential(0)}` : ''}
      </figcaption>
    </figure>
  )
}

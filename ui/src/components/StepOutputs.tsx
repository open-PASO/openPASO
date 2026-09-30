import { useEffect, useState } from 'react'
import { api, fileUrl } from '../api'
import type { FieldSeries } from '../types'
import Stage from './Stage'

/* What one solver step wrote into its own folder, shown under that step with that
   step's verdict.

   The pictures and the field used to sit above the whole transcript, taken newest
   first from anywhere in the run's folder. The transcript scrolls down as the run
   goes on, so they were out of sight; and a later run that drew nothing showed an
   earlier run's wake as the result. Measured 2026-09-29: the picture on top of the
   page came from a run whose drag was 53,288 at every step, with nothing to say so. */

export const PICTURE = /\.(png|jpe?g|svg|gif|webp)$/i

/** The step's folder inside this run's work folder, or null when it lies elsewhere:
    only this run's own folder is ever read. */
export function stepFolder(result: string, runId: string): string | null {
  const m = result.match(/\\?"work_dir\\?"\s*:\s*\\?"([^"\\]+)/)
  if (!m) return null
  const marker = `/webui_${runId}/work/`
  const at = m[1].indexOf(marker)
  return at < 0 ? null : m[1].slice(at + marker.length).replace(/\/+$/, '')
}

type Verdict = 'verified' | 'unverified' | 'failed'

const SAID: Record<Verdict, string> = {
  verified: 'openPASO verified this run',
  unverified: 'openPASO did NOT verify this run: this is what it wrote, not a result',
  failed: 'this run failed: this is what it left behind, not a result',
}

export default function StepOutputs({ runId, sub, verdict }: { runId: string; sub: string; verdict: Verdict }) {
  const [pics, setPics] = useState<{ rel: string; name: string }[]>([])
  const [field, setField] = useState<FieldSeries | null>(null)
  useEffect(() => {
    let dead = false
    api.files(runId, sub).then(async (d) => {
      const found: { rel: string; name: string }[] = []
      let series: FieldSeries | null = null
      for (const f of d.entries) {
        if (f.is_dir) continue
        if (PICTURE.test(f.name)) found.push({ rel: f.rel_path, name: f.name })
        else if (!series && f.name.endsWith('.json') && (f.size ?? 0) >= 200) {
          const v = await api.viz(f.rel_path).catch(() => null)
          if (v?.kind === 'field_series') series = v as unknown as FieldSeries
        }
      }
      if (!dead) { setPics(found); setField(series) }
    }).catch(() => { /* a folder that is gone shows nothing */ })
    return () => { dead = true }
  }, [runId, sub])
  if (!pics.length && !field) return null
  return (
    <section className="pl-7 mt-3" aria-label="What this run wrote">
      <div className={`text-[13px] font-medium mb-2 ${verdict === 'verified' ? 'text-muted' : 'text-coral'}`}>
        What this run wrote · {SAID[verdict]}
      </div>
      {field && <div className="mb-3"><Stage series={field} /></div>}
      {pics.length > 0 && (
        <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
          {pics.map((p) => (
            <figure key={p.rel} className="bg-soft border line rounded-[8px] p-3">
              <img src={fileUrl(p.rel)} alt="" loading="lazy" className="w-full rounded-[6px] bg-white" />
              <figcaption className="num mt-2 text-[13px] text-muted break-all">{sub}/{p.name}</figcaption>
            </figure>
          ))}
        </div>
      )}
    </section>
  )
}

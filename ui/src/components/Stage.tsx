import { useEffect, useRef, useState } from 'react'
import type { FieldSeries } from '../types'

/* The stage: a field the solver produced, one frame per stored timestep.

   Two ramps, chosen by the field's own range rather than by habit.

   A signed field (vorticity, a velocity component) diverges about zero: coral
   for one sign, graphite for the other, with the well showing through where the
   field is nothing. The two peaks are matched in luminance (0.470 against
   0.494) so neither sign visually outweighs the other.

   A field that never changes sign — temperature, pressure, a magnitude — is
   drawn on one rising ramp instead. Read on the diverging one, its low values
   sat in a dark "zero" well that means nothing here, so a cold region looked
   like the middle of a field with a sign, and the picture said something about
   the physics that the numbers do not. */
const NEG = [0xaf, 0xbc, 0xcb]
const ZERO = [0x08, 0x0b, 0x11]
const POS = [0xff, 0x9d, 0x82]
const LOW = [0x10, 0x16, 0x20]
const HOLE = [0x1d, 0x25, 0x30]

function ramp(signed: boolean) {
  const lut = new Uint8Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    if (signed) {
      const t = (i / 255) * 2 - 1
      const a = Math.pow(Math.abs(t), 1.1)
      const end = t < 0 ? NEG : POS
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(ZERO[c] + (end[c] - ZERO[c]) * a)
    } else {
      const a = Math.pow(i / 255, 0.9)
      for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(LOW[c] + (POS[c] - LOW[c]) * a)
    }
  }
  return lut
}

const b64 = (s: string) => {
  const bin = atob(s)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

type Data = {
  nx: number; ny: number
  mask: Uint8Array; frames: Uint8Array
  times: number[]; fps: number
}

export default function Stage({ series }: { series: FieldSeries }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const data = useRef<Data | null>(null)
  // signed when the field's own range crosses zero, which is what the writer
  // recorded; a field entirely on one side of zero is not a signed field
  const signed = (series.vmin ?? 0) < 0 && (series.vmax ?? 0) > 0
  const lut = useRef(ramp(signed))
  useEffect(() => { lut.current = ramp(signed) }, [signed])
  // the colour range the viewer chose; the frames stay as stored, only the colour lookup changes
  const [range, setRange] = useState<[number, number]>([series.vmin, series.vmax])
  const shown = useRef(lut.current)
  useEffect(() => {
    const [lo, hi] = range
    const out = new Uint8Array(256 * 3)
    for (let q = 0; q < 256; q++) {
      const v = series.vmin + (q / 255) * (series.vmax - series.vmin)
      const k = hi > lo ? Math.min(255, Math.max(0, Math.round(((v - lo) / (hi - lo)) * 255))) : q
      out.set(lut.current.subarray(k * 3, k * 3 + 3), q * 3)
    }
    shown.current = out
    draw(at.current)
  }, [range, signed, series.vmin, series.vmax])
  const raf = useRef(0)
  const [playing, setPlaying] = useState(true)
  const [n, setN] = useState(0)
  // the clock and the slider are written straight to the DOM while it plays:
  // React state for them re-rendered the whole stage on every frame
  const clockEl = useRef<HTMLSpanElement>(null)
  const slider = useRef<HTMLInputElement>(null)
  const at = useRef(0)

  // draw is deliberately not a hook dependency: it reads refs, so a redraw
  // never re-renders React.
  function draw(i: number) {
    const d = data.current
    const cv = canvas.current
    if (!d || !cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return
    const img = ctx.createImageData(d.nx, d.ny)
    const base = i * d.nx * d.ny
    for (let row = 0; row < d.ny; row++) {
      const src = (d.ny - 1 - row) * d.nx // the grid starts at the bottom, canvas at the top
      for (let col = 0; col < d.nx; col++) {
        const s = src + col
        const o = (row * d.nx + col) * 4
        if (!d.mask[s]) {
          img.data[o] = HOLE[0]; img.data[o + 1] = HOLE[1]
          img.data[o + 2] = HOLE[2]; img.data[o + 3] = 255
          continue
        }
        const v = d.frames[base + s] * 3
        img.data[o] = shown.current[v]
        img.data[o + 1] = shown.current[v + 1]
        img.data[o + 2] = shown.current[v + 2]
        img.data[o + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
    at.current = i
    if (clockEl.current) clockEl.current.textContent = `t = ${(d.times[i] ?? 0).toFixed(3)} s`
    if (slider.current && document.activeElement !== slider.current) slider.current.value = String(i)
  }

  useEffect(() => {
    let dead = false
    fetch(series.url)
      .then((r) => r.json())
      .then((m) => {
        if (dead) return
        data.current = {
          nx: m.nx, ny: m.ny,
          mask: b64(m.mask), frames: b64(m.frames),
          times: m.times, fps: m.fps || 25,
        }
        setN(m.times.length)
        const cv = canvas.current
        if (cv) { cv.width = m.nx; cv.height = m.ny }
        draw(0)
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) setPlaying(false)
      })
    return () => { dead = true; cancelAnimationFrame(raf.current) }
  }, [series.url])

  useEffect(() => {
    if (!playing || !n) return
    let last = 0
    let i = at.current
    const tick = (now: number) => {
      const d = data.current
      if (d && now - last >= 1000 / d.fps) {
        i = (i + 1) % n
        draw(i)                 // the canvas and the clock, without a re-render
        last = now
      }
      raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, n])

  const prov = series.provenance
  const pct = prov?.saturated_fraction != null
    ? `${(prov.saturated_fraction * 100).toFixed(1)}%` : 'some'
  const span = series.dx && series.dy
    ? `${(series.dx * series.nx).toFixed(2)} x ${(series.dy * series.ny).toFixed(2)} m`
    : ''

  return (
    <section className="bg-soft rounded-[12px] p-6" data-testid="stage">
      {/* Aspect comes from the data, and is driven from the width alone.

          Setting `height: min(420px, 42vw)` and then clamping the width with
          max-w-full pinned the height while the width hit the container, so a
          440x82 field of a 2.2 x 0.41 m channel rendered at 1280x420: a ratio
          of 3.05 where the physics says 5.37. The cylinder came out as an
          ellipse and vortex spacing measured off the picture was wrong by 76%
          in y. Width times the data's own ratio is the only correct height. */}
      <canvas ref={canvas} aria-label={series.field}
              className="block w-full"
              style={{ aspectRatio: `${series.nx} / ${series.ny}` }} />
      <div className="w-full mt-5 flex items-center gap-4">
        <button
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? 'Pause the field' : 'Play the field'}
          className="h-7 px-3 rounded-[6px] border line text-muted text-xs
                     transition-colors duration-150 hover:text-ink2"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
        <span ref={clockEl} className="num text-[13px] text-muted">t = 0.000 s</span>
        <input
          ref={slider}
          type="range" min={0} max={Math.max(0, n - 1)} defaultValue={0}
          aria-label="Frame"
          onChange={(e) => { setPlaying(false); draw(Number(e.target.value)) }}
          className="w-64 accent-coral"
        />
        <span className="ml-auto flex items-center gap-2.5 font-mono text-[13px] text-muted">
          <input type="number" step="any" value={range[0]} aria-label="Lowest value on the colour bar"
                 onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) setRange([v, range[1]]) }}
                 className="w-20 h-7 px-1.5 rounded-[6px] border line bg-transparent text-right" />
          <span className="w-[132px] h-1.5 rounded-sm"
                style={{ background: signed
                  ? 'linear-gradient(90deg,#AFBCCB,#64748B,#080B11,#C94A30,#FF9D82)'
                  : 'linear-gradient(90deg,#101620,#6B4A44,#C94A30,#FF9D82)' }} />
          <input type="number" step="any" value={range[1]} aria-label="Highest value on the colour bar"
                 onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) setRange([range[0], v]) }}
                 className="w-20 h-7 px-1.5 rounded-[6px] border line bg-transparent" />
          <span>{series.unit}</span>
          {(range[0] !== series.vmin || range[1] !== series.vmax) && (
            <button onClick={() => setRange([series.vmin, series.vmax])}
                    className="h-7 px-2 rounded-[6px] border line text-xs hover:text-ink2">Reset</button>
          )}
        </span>
      </div>

      {prov && (
        <p className="w-full mt-4 font-mono text-[13px] text-muted leading-relaxed">
          {span && `${span} · `}
          {series.nx} x {series.ny} grid, values at cell centres.
          {typeof prov.true_min === 'number' && typeof prov.true_max === 'number' && (
            <> Field reaches {prov.true_min.toFixed(1)} to {prov.true_max.toFixed(1)} {series.unit};
              colours span {series.vmin} to {series.vmax}
              {prov.clip_percentile ? ` (${prov.clip_percentile}th percentile)` : ''},
              so {pct} of the domain is saturated and reads as a bound, not a value.</>
          )}
          {prov.quantisation_step && <> {prov.quantisation_step.toFixed(3)} {series.unit} per colour level.</>}
          {prov.solver && <> {prov.solver}.</>}
        </p>
      )}
    </section>
  )
}

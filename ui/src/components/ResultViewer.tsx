import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { ABOVE, ABOVE_RGB, BELOW, BELOW_RGB, CSS, LUT, fmt, ticks, type Scale } from '../colours'

/* A result file in 3D, in the browser.

   The server sends the outer surface of the mesh and the values on it once (webui/mesh3d.py); every
   change here (quantity, colours, range, moved shape, step) is redrawn by the browser's own graphics
   card, so nothing is computed again and nothing costs anything. What the picture cannot show is
   said on it: only the outer surface is drawn, element values are averaged to the nodes, a moved
   shape is enlarged by the factor shown, and values outside the colour range have their own colours.

   Beside the picture, the person can say in plain words how it should look. That is turned into the
   same settings the controls write (webui/view_assist.py), shown in the controls and undoable; a
   request that needs a new calculation is handed to the run's own agent as a draft message, never
   run from here. */

type Field = {
  name: string; from: 'point' | 'cell'; ncomp: number; values: string
  min: number; max: number; body_min: number; body_max: number; body_min_c: number[]; body_max_c: number[]
}
type Payload = {
  name: string; rel: string; n_points: number; n_triangles: number; n_cells: number; cell_types: string[]
  bounds: [number[], number[]]; positions: string; triangles: string; fields: Field[]
  series: string[]; series_ranges: Record<string, { min: number; max: number; min_c: number[]; max_c: number[] }> | null
}
type Frame = { p: Payload; pos: Float32Array; tri: Uint32Array; vals: Map<string, Float32Array> }
type Part = 'size' | 'x' | 'y' | 'z'
export type ViewSettings = {
  field: string | null; part: Part; range: { auto: true } | { auto: false; min: number; max: number }
  scale: 'auto' | Scale; warp_field: string | null; warp_factor: number; edges: boolean; step: number
  same_range: boolean; cut_axis: 'none' | 'x' | 'y' | 'z'; cut_at: number; cut_flip: boolean
}
type Cut = { key: string; pos: Float32Array; vals: Map<string, { ncomp: number; a: Float32Array }>; n: number; note?: string }

const PARTS: Part[] = ['size', 'x', 'y', 'z']
const PART_WORD: Record<Part, string> = { size: 'size', x: 'x part', y: 'y part', z: 'z part' }

function b64<T extends Float32Array | Uint32Array>(s: string, C: new (b: ArrayBuffer) => T): T {
  const bin = atob(s)
  const u = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i)
  return new C(u.buffer)
}

function decode(p: Payload): Frame {
  const vals = new Map<string, Float32Array>()
  for (const f of p.fields) vals.set(f.name, b64(f.values, Float32Array))
  return { p, pos: b64(p.positions, Float32Array), tri: b64(p.triangles, Uint32Array), vals }
}

const stepKind = (series: string[]) =>
  series.length < 2 ? '' : series[0].toLowerCase().endsWith('.pvd') || /time|_t\d|step/i.test(series[0]) ? 'Time step'
    : /level|lvl|refine/i.test(series.join(' ')) ? 'Mesh level' : 'File'

/** The value drawn for each node, from the chosen quantity and part. */
function scalar(f: Field, a: Float32Array, part: Part): Float32Array {
  const n = a.length / f.ncomp
  const out = new Float32Array(n)
  if (f.ncomp === 1) return a
  const k = PARTS.indexOf(part) - 1
  for (let i = 0; i < n; i++) {
    if (k >= 0 && k < f.ncomp) { out[i] = a[i * f.ncomp + k]; continue }
    let s = 0
    for (let c = 0; c < f.ncomp; c++) s += a[i * f.ncomp + c] ** 2
    out[i] = Math.sqrt(s)
  }
  return out
}

function defaults(p: Payload): ViewSettings {
  const disp = p.fields.find((f) => f.ncomp >= 2 && /disp|deform|^u$|^d$/i.test(f.name))
  return {
    field: disp?.name ?? p.fields[0]?.name ?? null, part: 'size', range: { auto: true }, scale: 'auto',
    warp_field: disp?.name ?? null, warp_factor: 1, edges: false, step: 0, same_range: true,
    cut_axis: 'none', cut_at: 0.5, cut_flip: false,
  }
}

const AXES = ['x', 'y', 'z'] as const

export default function ResultViewer({ rel, runId, checkStatus, onClose, onAskAgent }: {
  rel: string; runId: string; checkStatus?: string; onClose: () => void; onAskAgent?: (text: string) => void
}) {
  const host = useRef<HTMLDivElement>(null)
  const three = useRef<{
    renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.PerspectiveCamera; controls: OrbitControls
    mesh: THREE.Mesh; edges: THREE.LineSegments; outline: THREE.LineSegments; cut: THREE.Mesh
  } | null>(null)
  const [cut, setCut] = useState<Cut | null>(null)
  const frames = useRef(new Map<string, Frame>())
  const [first, setFirst] = useState<Payload | null>(null)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [s, setS] = useState<ViewSettings | null>(null)
  const [playing, setPlaying] = useState(false)
  const [probe, setProbe] = useState<string>('')
  const [outside, setOutside] = useState({ below: 0, above: 0 })
  const [ask, setAsk] = useState('')
  const [talk, setTalk] = useState<{ you: string; said: string; need?: string | null; before?: ViewSettings; failed?: boolean }[]>([])
  const [asking, setAsking] = useState(false)
  const series = first?.series?.length ? first.series : first ? [first.rel] : []
  const kind = stepKind(series)

  // ── load the first file, then any step on demand ──
  const load = useCallback(async (r: string) => {
    const have = frames.current.get(r)
    if (have) return have
    const res = await fetch(`/api/mesh3d?rel=${encodeURIComponent(r)}`)
    const body = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error((body as { detail?: string }).detail || `${res.status}`)
    const f = decode(body as Payload)
    frames.current.set(r, f)
    return f
  }, [])
  useEffect(() => {
    let dead = false
    setLoading(true)
    load(rel).then((f) => {
      if (dead) return
      setFirst(f.p); setFrame(f)
      const start = Math.max(0, (f.p.series || []).indexOf(f.p.rel))
      setS({ ...defaults(f.p), step: start })
      setLoading(false)
    }).catch((e) => { if (!dead) { setErr(String(e.message || e)); setLoading(false) } })
    return () => { dead = true }
  }, [rel, load])
  useEffect(() => {
    if (!s || !series.length) return
    const r = series[Math.min(s.step, series.length - 1)]
    if (frame?.p.rel === r) return
    let dead = false
    setLoading(true)
    load(r).then((f) => { if (!dead) { setFrame(f); setLoading(false) } })
      .catch((e) => { if (!dead) { setErr(String(e.message || e)); setLoading(false) } })
    return () => { dead = true }
  }, [s?.step, series.join('|')])   // eslint-disable-line react-hooks/exhaustive-deps

  // ── the 3D scene, once ──
  useEffect(() => {
    const el = host.current
    if (!el) return
    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
    renderer.setClearColor(0x14181f)
    el.appendChild(renderer.domElement)
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(35, 1, 1e-6, 1e9)
    // soft, even light that follows the view: shading changes how bright a colour looks, so it stays mild
    scene.add(new THREE.AmbientLight(0xffffff, 1.6))
    const key = new THREE.DirectionalLight(0xffffff, 0.9)
    camera.add(key); key.position.set(1, 1, 2); scene.add(camera)
    const mesh = new THREE.Mesh(new THREE.BufferGeometry(),
      new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, flatShading: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }))
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 }))
    const outline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.25 }))
    const cutMesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, flatShading: true }))
    scene.add(mesh, edges, outline, cutMesh)
    renderer.localClippingEnabled = true
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
    three.current = { renderer, scene, camera, controls, mesh, edges, outline, cut: cutMesh }
    const size = () => {
      const w = el.clientWidth, h = el.clientHeight
      renderer.setSize(w, h); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix()
    }
    size()
    const ro = new ResizeObserver(size); ro.observe(el)
    let raf = 0
    const tick = () => { controls.update(); renderer.render(scene, camera); raf = requestAnimationFrame(tick) }
    tick()
    return () => { cancelAnimationFrame(raf); ro.disconnect(); controls.dispose(); renderer.dispose(); el.removeChild(renderer.domElement); three.current = null }
  }, [])

  const fitView = useCallback((how: 'reset' | 'front' | 'top' | 'side' = 'reset') => {
    const t = three.current, p = first
    if (!t || !p) return
    const [lo, hi] = p.bounds
    const c = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2)
    const r = Math.max(1e-9, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2)
    const dir = { reset: [1, 0.8, 1.6], front: [0, 0, 1], top: [0, 1, 0.0001], side: [1, 0, 0] }[how]
    const d = new THREE.Vector3(...dir).normalize().multiplyScalar(r / Math.tan((35 * Math.PI) / 360) * 1.15)
    t.camera.position.copy(c).add(d); t.camera.near = r / 1000; t.camera.far = r * 100
    t.camera.updateProjectionMatrix(); t.controls.target.copy(c); t.controls.update()
  }, [first])
  useEffect(() => { fitView('reset') }, [fitView])

  // ── what the colours mean: quantity, part, range, scale ──
  const field = frame && s?.field ? frame.p.fields.find((f) => f.name === s.field) ?? null : null
  const values = useMemo(() => field && frame && s ? scalar(field, frame.vals.get(field.name)!, field.ncomp > 1 ? s.part : 'size') : null,
    [field, frame, s?.part])  // eslint-disable-line react-hooks/exhaustive-deps
  const autoRange = useMemo((): [number, number] => {
    if (!field || !s) return [0, 1]
    const k = field.ncomp > 1 ? PARTS.indexOf(s.part) - 1 : -1
    const sr = s.same_range && first?.series_ranges?.[field.name]
    const src = sr ? { min: sr.min, max: sr.max, min_c: sr.min_c, max_c: sr.max_c }
      : { min: field.body_min, max: field.body_max, min_c: field.body_min_c, max_c: field.body_max_c }
    return k >= 0 ? [src.min_c[k], src.max_c[k]] : [src.min, src.max]
  }, [field, s?.part, s?.same_range, first])  // eslint-disable-line react-hooks/exhaustive-deps
  const signed = autoRange[0] < 0 && autoRange[1] > 0
  const scale: Scale = s?.scale === 'auto' || !s ? (signed ? 'diverging' : 'sequential') : s.scale
  let [lo, hi] = s && !s.range.auto ? [s.range.min, s.range.max] : autoRange
  if (s?.range.auto && scale === 'diverging') { const m = Math.max(Math.abs(lo), Math.abs(hi)); lo = -m; hi = m }
  if (!(hi > lo)) hi = lo + (Math.abs(lo) || 1) * 1e-6

  // ── draw ──
  const cutK = s && s.cut_axis !== 'none' ? AXES.indexOf(s.cut_axis) : -1
  const cutPos = first && s && cutK >= 0 ? first.bounds[0][cutK] + s.cut_at * (first.bounds[1][cutK] - first.bounds[0][cutK]) : null
  useEffect(() => {
    if (!frame || cutK < 0 || cutPos == null) { setCut(null); return }
    const key = `${frame.p.rel}|${cutK}|${cutPos}`
    let dead = false
    const t = window.setTimeout(async () => {
      const r = await fetch(`/api/mesh3d/cut?rel=${encodeURIComponent(frame.p.rel)}&axis=${cutK}&at=${cutPos}`)
      const b = await r.json().catch(() => ({}))
      if (dead) return
      if (!r.ok) { setCut({ key, pos: new Float32Array(), vals: new Map(), n: 0, note: b.detail || String(r.status) }); return }
      const vals = new Map<string, { ncomp: number; a: Float32Array }>()
      for (const f of b.fields as { name: string; ncomp: number; values: string }[]) vals.set(f.name, { ncomp: f.ncomp, a: b64(f.values, Float32Array) })
      setCut({ key, pos: b.positions ? b64(b.positions, Float32Array) : new Float32Array(), vals, n: b.n_triangles, note: b.note })
    }, 250)
    return () => { dead = true; window.clearTimeout(t) }
  }, [frame, cutK, cutPos])

  useEffect(() => {
    const t = three.current
    if (!t || !frame || !s) return
    const n = frame.pos.length / 3
    const pos = new Float32Array(frame.pos)
    const wf = s.warp_field ? frame.p.fields.find((f) => f.name === s.warp_field) : null
    if (wf && wf.ncomp >= 2 && s.warp_factor !== 0) {
      const a = frame.vals.get(wf.name)!
      for (let i = 0; i < n; i++) for (let c = 0; c < Math.min(3, wf.ncomp); c++) pos[i * 3 + c] += s.warp_factor * a[i * wf.ncomp + c]
    }
    const col = new Float32Array(n * 3)
    let below = 0, above = 0
    const lut = LUT[scale]
    for (let i = 0; i < n; i++) {
      if (!values) { col.set([0.62, 0.66, 0.72], i * 3); continue }
      const v = values[i]
      if (!Number.isFinite(v)) { col.set([0.3, 0.3, 0.3], i * 3); continue }
      if (v < lo) { col.set(BELOW_RGB, i * 3); below++; continue }
      if (v > hi) { col.set(ABOVE_RGB, i * 3); above++; continue }
      const k = Math.round(((v - lo) / (hi - lo)) * 255) * 3
      col[i * 3] = lut[k]; col[i * 3 + 1] = lut[k + 1]; col[i * 3 + 2] = lut[k + 2]
    }
    setOutside({ below: below / n, above: above / n })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('color', new THREE.BufferAttribute(col, 3))
    g.setIndex(new THREE.BufferAttribute(frame.tri, 1))
    g.computeVertexNormals()
    t.mesh.geometry.dispose(); t.mesh.geometry = g
    t.edges.geometry.dispose()
    t.edges.geometry = s.edges && frame.tri.length / 3 <= 400_000 ? new THREE.WireframeGeometry(g) : new THREE.BufferGeometry()
    // the shape before it moved, faintly, whenever the drawn shape is moved
    t.outline.geometry.dispose()
    if (wf && s.warp_factor !== 0) {
      const g0 = new THREE.BufferGeometry()
      g0.setAttribute('position', new THREE.BufferAttribute(frame.pos, 3)); g0.setIndex(new THREE.BufferAttribute(frame.tri, 1))
      t.outline.geometry = new THREE.EdgesGeometry(g0, 30)
      g0.dispose()
    } else t.outline.geometry = new THREE.BufferGeometry()
    // the cut: the outer surface is opened at the plane, and the plane shows the values inside
    const planes = cutK >= 0 && cutPos != null
      ? [new THREE.Plane(new THREE.Vector3(...[0, 1, 2].map((k) => (k === cutK ? (s.cut_flip ? 1 : -1) : 0)) as [number, number, number]), s.cut_flip ? -cutPos : cutPos)]
      : []
    for (const o of [t.mesh, t.edges, t.outline]) (o.material as THREE.Material).clippingPlanes = planes
    t.cut.geometry.dispose()
    if (planes.length && cut && cut.n > 0) {
      const cp = new Float32Array(cut.pos)
      const m = cp.length / 3
      const cw = wf && s.warp_factor !== 0 ? cut.vals.get(wf.name) : undefined
      if (cw) for (let i = 0; i < m; i++) for (let c = 0; c < Math.min(3, cw.ncomp); c++) cp[i * 3 + c] += s.warp_factor * cw.a[i * cw.ncomp + c]
      const cv = field ? cut.vals.get(field.name) : undefined
      const v = cv && field ? scalar(field, cv.a, field.ncomp > 1 ? s.part : 'size') : null
      const cc = new Float32Array(m * 3)
      for (let i = 0; i < m; i++) {
        if (!v) { cc.set([0.62, 0.66, 0.72], i * 3); continue }
        const x = v[i]
        if (!Number.isFinite(x)) cc.set([0.3, 0.3, 0.3], i * 3)
        else if (x < lo) cc.set(BELOW_RGB, i * 3)
        else if (x > hi) cc.set(ABOVE_RGB, i * 3)
        else { const k = Math.round(((x - lo) / (hi - lo)) * 255) * 3; cc[i * 3] = lut[k]; cc[i * 3 + 1] = lut[k + 1]; cc[i * 3 + 2] = lut[k + 2] }
      }
      const g2 = new THREE.BufferGeometry()
      g2.setAttribute('position', new THREE.BufferAttribute(cp, 3)); g2.setAttribute('color', new THREE.BufferAttribute(cc, 3))
      g2.computeVertexNormals()
      t.cut.geometry = g2
    } else t.cut.geometry = new THREE.BufferGeometry()
  }, [frame, s, values, lo, hi, scale, cut, cutK, cutPos, field])

  // ── value under the pointer ──
  const onMove = (e: React.PointerEvent) => {
    const t = three.current
    if (!t || !values || !frame || e.buttons) return
    const r = t.renderer.domElement.getBoundingClientRect()
    const ray = new THREE.Raycaster()
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), t.camera)
    const hit = ray.intersectObject(t.mesh)[0]
    if (!hit?.face) { setProbe(''); return }
    const pos = t.mesh.geometry.getAttribute('position')
    const pt = hit.point
    let best = hit.face.a, bd = Infinity
    for (const k of [hit.face.a, hit.face.b, hit.face.c]) {
      const d = (pos.getX(k) - pt.x) ** 2 + (pos.getY(k) - pt.y) ** 2 + (pos.getZ(k) - pt.z) ** 2
      if (d < bd) { bd = d; best = k }
    }
    setProbe(`${fmt(values[best])} ${field?.from === 'cell' ? '(element values averaged to this node)' : 'at this node'}`)
  }

  // ── play through the steps (no blending: different meshes are not moments in between) ──
  useEffect(() => {
    if (!playing || series.length < 2) return
    const id = window.setInterval(() => setS((v) => v && { ...v, step: (v.step + 1) % series.length }), 1200)
    return () => window.clearInterval(id)
  }, [playing, series.length])

  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onClose() } }
    document.addEventListener('keydown', k, true)
    return () => document.removeEventListener('keydown', k, true)
  }, [onClose])

  const quantity = field ? `${field.name}${field.ncomp > 1 ? `, ${PART_WORD[s!.part]}` : ''}` : 'shape only'
  const wfield = s?.warp_field ? frame?.p.fields.find((f) => f.name === s.warp_field) : null
  const moved = wfield && s ? `Shape moved by ${wfield.name} ×${fmt(s.warp_factor)}${s.warp_factor === 1 ? ' (true size)' : ' (enlarged)'}` : ''
  const stepLine = series.length > 1 && s ? `${kind} ${s.step + 1} of ${series.length}` : ''
  const fileName = frame?.p.name ?? rel.split('/').pop()

  const save = () => {
    const t = three.current
    if (!t || !s) return
    const src = t.renderer.domElement
    const W = src.width, H = src.height, pad = Math.round(H * 0.03), fs = Math.max(14, Math.round(H * 0.022))
    const cv = document.createElement('canvas'); cv.width = W + Math.round(W * 0.14); cv.height = H + fs * 6
    const g = cv.getContext('2d')!
    g.fillStyle = '#14181f'; g.fillRect(0, 0, cv.width, cv.height); g.drawImage(src, 0, 0)
    // colour bar with its numbers
    const bx = W + pad, by = pad * 2, bh = H - pad * 4, bw = Math.round(W * 0.025)
    const lut = LUT[scale]
    for (let y = 0; y < bh; y++) {
      const k = Math.round((1 - y / bh) * 255) * 3
      g.fillStyle = `rgb(${lut[k] * 255},${lut[k + 1] * 255},${lut[k + 2] * 255})`; g.fillRect(bx, by + y, bw, 1)
    }
    g.fillStyle = ABOVE; g.fillRect(bx, by - fs * 0.8, bw, fs * 0.6); g.fillStyle = BELOW; g.fillRect(bx, by + bh + fs * 0.2, bw, fs * 0.6)
    g.fillStyle = '#e6e8eb'; g.font = `${fs}px sans-serif`
    for (const v of ticks(lo, hi)) g.fillText(fmt(v), bx + bw + 6, by + bh * (1 - (v - lo) / (hi - lo)) + fs * 0.35)
    const lines = [`${fileName}${stepLine ? ' · ' + stepLine : ''} · ${quantity} (units as in the file)`,
      `Colours ${fmt(lo)} to ${fmt(hi)}; below: dark grey, above: magenta${moved ? ' · ' + moved : ''}`,
      `${cutPos != null ? `Cut at ${s.cut_axis} = ${fmt(cutPos)} (cut face: values inside)` : 'Outer surface only'}${field?.from === 'cell' ? ' · element values averaged to nodes' : ''} · drawn by the openPASO viewer from this run's own file`,
      checkStatus ? `openPASO check of this run: ${checkStatus}` : 'openPASO check of this run: see the run page']
    lines.forEach((l, i) => g.fillText(l, pad, H + fs * (1.4 + i * 1.2)))
    const a = document.createElement('a')
    a.download = `${(fileName || 'result').replace(/\.[^.]+$/, '')}_view.png`; a.href = cv.toDataURL('image/png'); a.click()
  }

  const send = async () => {
    const text = ask.trim()
    if (!text || !s || !frame) return
    setAsking(true)
    try {
      const r = await fetch(`/api/sessions/${runId}/view-assist`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, steps: series.length, step_kind: kind,
          fields: frame.p.fields.map((f) => ({ name: f.name, ncomp: f.ncomp })), current: s }),
      })
      const b = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(b.detail || `${r.status}`)
      const before = s
      const next = { ...s, ...(b.settings as Partial<ViewSettings>) }
      const changed = Object.keys(b.settings || {}).length > 0
      if (changed) setS(next)
      if ((b.settings || {}).view) fitView(b.settings.view)
      setTalk((h) => [...h, { you: text, said: b.said, need: b.needs_calculation, before: changed ? before : undefined }])
      setAsk('')
    } catch (e) {
      setTalk((h) => [...h, { you: text, said: `No answer: ${String((e as Error).message || e)}`, failed: true }])
    } finally { setAsking(false) }
  }

  const set = (p: Partial<ViewSettings>) => setS((v) => v && { ...v, ...p })
  const fields = frame?.p.fields ?? []
  const vectors = fields.filter((f) => f.ncomp >= 2)

  return (
    <div className="fixed inset-0 z-40 bg-canvas flex flex-col" role="dialog" aria-modal="true" aria-label={`3D view of ${fileName}`}>
      <header className="h-12 px-5 border-b line flex items-center gap-4 shrink-0">
        <span className="num text-[14px] text-ink2 truncate">{fileName}</span>
        <span className="text-[13px] text-muted">3D view of a result file</span>
        <button autoFocus onClick={onClose} className="ml-auto h-8 px-3.5 rounded-[6px] border line text-[13px] text-muted hover:text-ink2">Back to the run</button>
      </header>
      <div className="flex-1 min-h-0 flex">
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="relative flex-1 min-h-0" ref={host} onPointerMove={onMove} onPointerLeave={() => setProbe('')}>
            <div className="absolute left-4 top-3 text-[13px] leading-[1.6] text-ink2 pointer-events-none" aria-live="polite">
              <div>{quantity}{field ? ' · units as written in the file' : ''}</div>
              <div className="text-muted">{cutPos != null
                ? `Cut at ${s?.cut_axis} = ${fmt(cutPos)}: the cut face shows the values inside the body`
                : 'Outer surface only: values inside the body are not drawn'}{field?.from === 'cell' ? ' · element values averaged to nodes' : ''}</div>
              {moved && <div className={s?.warp_factor === 1 ? 'text-muted' : 'text-coral'}>{moved}</div>}
              {probe && <div className="num">Value here: {probe}</div>}
            </div>
            {field && (
              <div className="absolute right-4 top-12 bottom-12 w-24 flex flex-col items-start pointer-events-none" aria-label="Colour scale">
                <div className="num text-[12px] text-muted mb-1">above {outside.above > 0 ? `(${(outside.above * 100).toFixed(1)} %)` : ''}</div>
                <div className="flex flex-1 gap-2 min-h-0 w-full">
                  <div className="w-3.5 flex flex-col">
                    <div className="h-2.5 mb-0.5" style={{ background: ABOVE }} />
                    <div className="flex-1 rounded-sm" style={{ background: CSS[scale] }} />
                    <div className="h-2.5 mt-0.5" style={{ background: BELOW }} />
                  </div>
                  <div className="relative flex-1 num text-[12px] text-ink2">
                    {ticks(lo, hi).map((v) => (
                      <span key={v} className="absolute -translate-y-1/2" style={{ top: `${(1 - (v - lo) / (hi - lo)) * 100}%` }}>{fmt(v)}</span>
                    ))}
                  </div>
                </div>
                <div className="num text-[12px] text-muted mt-1">below {outside.below > 0 ? `(${(outside.below * 100).toFixed(1)} %)` : ''}</div>
              </div>
            )}
            {loading && <div className="absolute inset-0 grid place-items-center text-[14px] text-muted">Reading the result file…</div>}
            {err && <div className="absolute inset-0 grid place-items-center p-10 text-center text-[15px] text-bad">openPASO could not show this file: {err}</div>}
          </div>
          {series.length > 1 && s && (
            <div className="h-14 px-5 border-t line flex items-center gap-4 shrink-0">
              <button onClick={() => setPlaying((p) => !p)} className="h-8 px-3 rounded-[6px] border line text-[13px] text-muted hover:text-ink2">{playing ? 'Pause' : 'Play'}</button>
              <span className="text-[13px] text-ink2 w-36">{stepLine}</span>
              <input type="range" min={0} max={series.length - 1} value={s.step} aria-label={kind}
                     onChange={(e) => { setPlaying(false); set({ step: Number(e.target.value) }) }} className="flex-1 accent-coral" />
              <span className="num text-[12px] text-muted truncate max-w-[260px]">{frame?.p.name}</span>
            </div>
          )}
        </div>

        <aside className="w-[340px] shrink-0 border-l line flex flex-col min-h-0">
          {s && (
            <section className="p-5 border-b line flex flex-col gap-3.5 text-[13px]" aria-label="View settings">
              <div className="text-[13px] font-semibold text-ink2">View settings</div>
              <label className="flex flex-col gap-1">
                <span className="text-muted">Colour by</span>
                <div className="flex gap-2">
                  <select value={s.field ?? ''} onChange={(e) => set({ field: e.target.value || null, part: 'size', range: { auto: true } })}
                          className="flex-1 h-8 px-2 rounded-[6px] border line bg-card text-ink2">
                    <option value="">nothing (shape only)</option>
                    {fields.map((f) => <option key={f.name} value={f.name}>{f.name}{f.ncomp > 1 ? ` (vector)` : ''}</option>)}
                  </select>
                  {field && field.ncomp > 1 && (
                    <select value={s.part} onChange={(e) => set({ part: e.target.value as Part, range: { auto: true } })}
                            className="w-24 h-8 px-2 rounded-[6px] border line bg-card text-ink2" aria-label="Which part">
                      {PARTS.slice(0, field.ncomp + 1).map((p) => <option key={p} value={p}>{PART_WORD[p]}</option>)}
                    </select>
                  )}
                </div>
              </label>
              {field && (
                <div className="flex flex-col gap-1">
                  <span className="text-muted">Colours from … to …</span>
                  <div className="flex items-center gap-2">
                    <input type="number" step="any" aria-label="Lowest colour value" value={Number(lo.toPrecision(6))}
                           onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) set({ range: { auto: false, min: v, max: hi } }) }}
                           className="w-24 h-8 px-2 rounded-[6px] border line bg-transparent num" />
                    <input type="number" step="any" aria-label="Highest colour value" value={Number(hi.toPrecision(6))}
                           onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) set({ range: { auto: false, min: lo, max: v } }) }}
                           className="w-24 h-8 px-2 rounded-[6px] border line bg-transparent num" />
                    <label className="flex items-center gap-1.5 text-muted">
                      <input type="checkbox" checked={s.range.auto} onChange={(e) => set({ range: e.target.checked ? { auto: true } : { auto: false, min: lo, max: hi } })} />Auto
                    </label>
                  </div>
                  {series.length > 1 && (
                    <label className="flex items-center gap-1.5 text-muted">
                      <input type="checkbox" checked={s.same_range} onChange={(e) => set({ same_range: e.target.checked })} />Same range for every {kind.toLowerCase()}
                    </label>
                  )}
                </div>
              )}
              {vectors.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-muted">Move the shape by</span>
                  <div className="flex items-center gap-2">
                    <select value={s.warp_field ?? ''} onChange={(e) => set({ warp_field: e.target.value || null })}
                            className="flex-1 h-8 px-2 rounded-[6px] border line bg-card text-ink2">
                      <option value="">nothing</option>
                      {vectors.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
                    </select>
                    {s.warp_field && (
                      <select value={[1, 10, 100].includes(s.warp_factor) ? String(s.warp_factor) : 'other'} aria-label="How much larger"
                              onChange={(e) => e.target.value !== 'other' && set({ warp_factor: Number(e.target.value) })}
                              className="w-28 h-8 px-2 rounded-[6px] border line bg-card text-ink2">
                        <option value="1">×1 true size</option><option value="10">×10</option><option value="100">×100</option>
                        <option value="other">×{fmt(s.warp_factor)}</option>
                      </select>
                    )}
                  </div>
                </div>
              )}
              <div className="flex flex-col gap-1">
                <span className="text-muted">Cut through the inside</span>
                <div className="flex items-center gap-2">
                  <select value={s.cut_axis} onChange={(e) => set({ cut_axis: e.target.value as ViewSettings['cut_axis'] })}
                          className="w-28 h-8 px-2 rounded-[6px] border line bg-card text-ink2" aria-label="Cut direction">
                    <option value="none">no cut</option>
                    {AXES.map((a) => <option key={a} value={a}>across {a}</option>)}
                  </select>
                  {s.cut_axis !== 'none' && (
                    <>
                      <input type="range" min={0} max={1} step={0.01} value={s.cut_at} aria-label="Where to cut"
                             onChange={(e) => set({ cut_at: Number(e.target.value) })} className="flex-1 accent-coral" />
                      <button onClick={() => set({ cut_flip: !s.cut_flip })} title="Keep the other side"
                              className="h-7 px-2 rounded-[6px] border line text-[12px] text-muted hover:text-ink2">flip</button>
                    </>
                  )}
                </div>
                {cutPos != null && (
                  <span className="num text-[12px] text-muted">{s.cut_axis} = {fmt(cutPos)}{cut?.note ? ` · ${cut.note}` : ''}</span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <label className="flex items-center gap-1.5 text-muted mr-2">
                  <input type="checkbox" checked={s.edges} onChange={(e) => set({ edges: e.target.checked })} />Mesh lines
                </label>
                {(['reset', 'front', 'top', 'side'] as const).map((v) => (
                  <button key={v} onClick={() => fitView(v)} className="h-7 px-2.5 rounded-[6px] border line text-[12px] text-muted hover:text-ink2">
                    {v === 'reset' ? 'Reset view' : v}
                  </button>
                ))}
              </div>
              <button onClick={save} className="h-8 rounded-[6px] border line text-[13px] text-ink2 hover:border-strong">Save picture (with its labels)</button>
              <p className="text-[12px] text-faint leading-[1.5]">
                {frame ? `${frame.p.n_triangles.toLocaleString()} surface triangles from ${frame.p.n_cells.toLocaleString()} cells (${frame.p.cell_types.join(', ')}).` : ''}
              </p>
            </section>
          )}
          <section className="flex-1 min-h-0 flex flex-col p-5" aria-label="Ask for a different view">
            <div className="text-[13px] font-semibold text-ink2">Ask for a different view</div>
            <p className="text-[12px] text-muted mt-0.5">Changes only how this file is shown. It does not talk to the run's agent and changes no result.</p>
            <ol className="flex-1 min-h-0 overflow-y-auto scroll mt-3 flex flex-col gap-3 text-[13px]">
              {talk.map((m, i) => (
                <li key={i}>
                  <div className="text-ink2">{m.you}</div>
                  <div className={`mt-1 ${m.failed ? 'text-bad' : 'text-body'}`}>{m.said}
                    {m.before && <button onClick={() => setS(m.before!)} className="ml-2 underline underline-offset-4 text-muted hover:text-ink">Undo</button>}
                  </div>
                  {m.need && (
                    <div className="mt-1.5 text-muted">Needs a new calculation: {m.need}
                      {onAskAgent && (
                        <button onClick={() => onAskAgent(`Please make this picture from ${fileName}: ${m.you}`)}
                                className="block mt-1.5 h-7 px-2.5 rounded-[6px] border line text-[12px] text-ink2 hover:border-strong">Write it as a message to the run's agent</button>
                      )}
                    </div>
                  )}
                </li>
              ))}
              {!talk.length && (
                <li className="text-faint">For example: "colour by pressure from 0 to 50", "show the bending ten times larger", "same colours for every mesh level".</li>
              )}
            </ol>
            <form className="mt-3 flex gap-2" onSubmit={(e) => { e.preventDefault(); void send() }}>
              <input value={ask} onChange={(e) => setAsk(e.target.value)} placeholder="How should it look?" aria-label="How should the picture look"
                     className="flex-1 h-9 px-3 rounded-[8px] border line bg-transparent text-[14px]" disabled={asking} />
              <button disabled={asking || !ask.trim()} className="h-9 px-3 rounded-[8px] bg-coral text-on-coral text-[13px] font-semibold disabled:opacity-50">{asking ? '…' : 'Apply'}</button>
            </form>
          </section>
        </aside>
      </div>
    </div>
  )
}

/* Colours for result pictures, chosen by what the numbers are, never a rainbow.

   A quantity that only grows (a size, a pressure above zero) gets a one-direction scale (viridis);
   one that has a sign gets a two-direction scale centred on zero (blue - white - orange). Values
   outside the chosen range get their own colours, so a clipped peak never passes for the top of the
   scale: dark grey below, magenta above. Both scales stay readable for colour-blind readers. */

export type Scale = 'sequential' | 'diverging'

const VIRIDIS = ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725']
const BWO = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fee0b6', '#fdb863', '#e08214', '#b35806']
export const BELOW = '#2b2f36'
export const ABOVE = '#ff3bd4'

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)

function table(stops: string[]): Float32Array {
  const rgb = stops.map(hex)
  const lut = new Float32Array(256 * 3)
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1)
    const k = Math.min(rgb.length - 2, Math.floor(t))
    const f = t - k
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = rgb[k][c] * (1 - f) + rgb[k + 1][c] * f
  }
  return lut
}

export const LUT: Record<Scale, Float32Array> = { sequential: table(VIRIDIS), diverging: table(BWO) }
export const CSS: Record<Scale, string> = {
  sequential: `linear-gradient(to top,${VIRIDIS.join(',')})`,
  diverging: `linear-gradient(to top,${BWO.join(',')})`,
}
export const BELOW_RGB = hex(BELOW)
export const ABOVE_RGB = hex(ABOVE)

/** About five round numbers between lo and hi, for the colour bar. */
export function ticks(lo: number, hi: number): number[] {
  if (!(hi > lo)) return [lo]
  const raw = (hi - lo) / 4
  const p = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v)
  return out
}

export function fmt(v: number): string {
  if (!Number.isFinite(v)) return '—'
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2).replace('e+', 'e')
  return String(Number(v.toPrecision(4)))
}

import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import { api } from '../api'
import { heroIn } from '../motion'
import { navigate } from '../route'
import type { AppConfig, ModelGroup } from '../types'
import Composer, { ModePicker, ModelPicker, findModel } from './Composer'
import Logo from './Logo'
import { runsChanged } from './Shell'

const EXAMPLES: [string, string][] = [
  ['Cylinder wake at Re 100', 'Flow past a cylinder at Reynolds 100 in a 2.2 x 0.41 channel. Show me the wake and report drag, lift and Strouhal number.'],
  ['Flap in a flow, two codes', 'A rubber flap (0.1 m wide, 0.5 m tall, E = 3.5 MPa, Poisson ratio 0.4) stands upright in a 3 m long, 1 m high water channel, clamped to the floor 1 m from the inlet. A viscous liquid (density 1000 kg/m³, viscosity 50 Pa·s) enters at a mean speed of 1 m/s. How far does the flap tip bend? Compute the flow and the flap with two different solvers, coupled by openPASO.'],
  ['Copper sheet from DFT', 'How stiff is a copper sheet with a square pattern of round holes (10 % porosity) under uniaxial tension? Take the elastic constants from the quantum-mechanical calculations in the attached hand-off card, but only if the calculations behind them are complete.'],
  ['Heat through a steel stud', 'Steady heat conduction through a timber wall with a steel stud. Show the temperature field and give the U-value.'],
  ['Cantilever under load', 'A steel cantilever beam, 1 m long, 50 x 50 mm, with 1 kN at the tip. Compare the tip deflection with beam theory.'],
]

function pref(key: string): string | null {
  try { return localStorage.getItem(key) } catch { return null }
}
function setPref(key: string, v: string) {
  try { localStorage.setItem(key, v) } catch { /* private window */ }
}

export default function Home({ config, groups }: { config: AppConfig | null; groups: ModelGroup[] | null }) {
  const [model, setModel] = useState<string | null>(null)
  // an older version stored "autonomous", which no longer exists: read back
  // verbatim it would be shown as the choice and refused when the run starts
  const [mode, setMode] = useState<string>(() => {
    const kept = pref('openpaso.mode')
    return kept === 'plan' || kept === 'accept' ? kept : 'accept'
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [solvers, setSolvers] = useState<{ ready: number; total: number } | null>(null)

  // Your model choice is remembered. It used to reset to the server default on
  // every new run, silently, so a run could use a model nobody chose.
  useEffect(() => {
    if (!groups) return
    const saved = pref('openpaso.model')
    const usable = findModel(groups, saved)
    setModel(usable?.available ? saved : (groups.flatMap((g) => g.models).find((m) => m.available)?.id ?? null))
  }, [groups])

  useEffect(() => {
    api.solvers().then((d) => d.ok && setSolvers({
      ready: d.solvers.filter((s) => s.status === 'available').length, total: d.solvers.length,
    })).catch(() => {})
  }, [])

  const cur = findModel(groups, model)
  const planAllowed = cur?.kind !== 'claude-code'
  useEffect(() => { if (!planAllowed && mode === 'plan') setMode('accept') }, [planAllowed, mode])

  async function start(text: string, files: File[]) {
    if (!model || !cur?.available) { setError('Choose a model that is available.'); return false }
    setBusy(true); setError(null)
    try {
      const s = await api.createRun(model, mode)
      let names: string[] = []
      if (files.length) {
        try {
          names = (await api.upload(s.id, files)).saved.map((x) => x.name)
        } catch (e) {
          // the run exists but was never prompted, so it is hidden from the
          // list: leaving it would leave a folder nobody can reach or remove
          await api.deleteRun(s.id).catch(() => {})
          throw e
        }
      }
      try {
        await api.startRun(s.id, text, names)
      } catch (e) {
        await api.deleteRun(s.id).catch(() => {})
        throw e
      }
      runsChanged()
      navigate({ run: s.id })
      return true
    } catch (e) {
      setError(String((e as Error).message || e))
      return false
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto scroll">
      <div className="min-h-full flex flex-col justify-center py-16">
        <div className="w-full max-w-[920px] mx-auto px-10">
          <motion.div {...heroIn(0)} className="flex justify-center">
            <Logo size={168} dim={draft.length > 0} />
          </motion.div>
          <motion.h1 {...heroIn(1)} className="mt-8 text-center text-[64px] font-semibold leading-[1.05] tracking-[-0.03em] text-ink">
            Describe the physics.
          </motion.h1>
          <motion.p {...heroIn(2)} className="mt-5 text-center text-[20px] leading-[1.5] text-body">
            openPASO picks a solver, writes its input, runs it, and shows you every step.
          </motion.p>

          <motion.div {...heroIn(3)} className="mt-10">
            <Composer
              placeholder="Describe a simulation: the geometry, the physics, and what you want to know"
              submitLabel="Run" onSubmit={start} busy={busy} autoFocus initial={draft} draftKey="home"
              left={<>
                <ModelPicker groups={groups} value={model} up={false}
                             onChange={(id) => { setModel(id); setPref('openpaso.model', id) }} />
                <ModePicker modes={config?.modes} value={mode} planAllowed={planAllowed} up={false}
                            onChange={(m) => { setMode(m); setPref('openpaso.mode', m) }} />
              </>} />
            {groups && !groups.some((g) => g.models.some((m) => m.available)) && (
              <div role="alert" className="mt-4 rounded-[8px] border border-bad/40 bg-bad/[0.06] px-5 py-4 text-[15px] leading-[1.55] text-ink2">
                <p className="text-ink font-medium">No model is available yet, so nothing can run.</p>
                <p className="mt-1">Do one of these, then reload this page:</p>
                <ul className="mt-1 list-disc pl-5">
                  <li>Install Claude Code and sign in once by running <code className="num">claude</code> in a terminal.</li>
                  <li>Add an OpenRouter key: put the line <code className="num">OPENROUTER_API_KEY=...</code> in a file named <code className="num">.env</code> in the openPASO folder.</li>
                  <li>Start a local model server on one of the ports listed under Model.</li>
                </ul>
              </div>
            )}
            {error && <p role="alert" className="mt-3 text-[14px] text-bad">{error}</p>}
          </motion.div>

          <motion.div {...heroIn(4)} className="mt-5 flex flex-wrap items-center gap-2">
            <span className="text-[14px] text-muted mr-1">Try</span>
            {EXAMPLES.map(([label, full]) => (
              <button key={label} type="button" onClick={() => setDraft(full)}
                      className="h-9 px-3.5 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong transition-colors">
                {label}
              </button>
            ))}
            <a href="/?view=solvers" onClick={(e) => { e.preventDefault(); navigate({ view: 'solvers' }) }}
               className="ml-auto text-[14px] text-muted hover:text-ink transition-colors">
              {solvers ? `${solvers.ready} of ${solvers.total} solvers found on this machine` : 'Checking solvers…'}
              <span aria-hidden className="ml-1">→</span>
            </a>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

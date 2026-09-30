import { useEffect, useMemo, useRef, useState } from 'react'
import { api, clock, fileUrl, money } from '../api'
import { navigate } from '../route'
import type { AppConfig, FieldSeries, ModelGroup } from '../types'
import { useRun } from '../useRun'
import Composer, { ModePicker, findModel } from './Composer'
import FilesDrawer from './FilesDrawer'
import Glyph, { STATE } from './Glyph'
import { runsChanged } from './Shell'
import Stage from './Stage'
import { stepFolder } from './StepOutputs'
import Transcript from './Transcript'

// the tools whose reply can be a solver result (webui/outcome.py SOLVER_TOOLS)
const SOLVER_TOOLS = new Set(['run_simulation', 'run_with_generator', 'coupled_solve', 'couple',
                              'couple_levels', 'couple_precice', 'verify_mesh_independence'])

function pref(key: string, fallback: string) {
  try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
}

/** The run's own field animation, if it wrote one. Only this run's folder is
    searched: showing a result from anywhere else is how a crashed run once
    displayed somebody else's vortex street as its answer. */
function useField(runId: string, settle: number) {
  const [field, setField] = useState<FieldSeries | null>(null)
  const [fieldAt, setFieldAt] = useState('')
  const [pictures, setPictures] = useState<{ rel: string; name: string; mtime: number }[]>([])
  useEffect(() => { setField(null); setFieldAt(''); setPictures([]) }, [runId])
  useEffect(() => {
    let dead = false
    // One scan, a little after the steps stop arriving. It used to start a
    // whole recursive walk of the run's folder on every tool result, so a long
    // run spent more requests looking for a picture than doing the work, and
    // several walks overlapped.
    let timer = 0
    // each scan keeps its own record of what it has looked at. A shared one let
    // a scan that was replaced mark a file as seen, so the scan that followed
    // skipped it and a field the run really wrote vanished from the page.
    const seen = new Map<string, number>()
    const pics: { rel: string; name: string; mtime: number }[] = []
    // the whole folder is walked, even after a field is found: stopping there
    // left out every picture that came after it
    let found: FieldSeries | null = null
    let foundAt = ''
    const walk = async (sub: string, depth: number): Promise<void> => {
      if (depth > 3) return
      const d = await api.files(runId, sub).catch(() => null)
      if (!d) return
      for (const f of d.entries) {
        if (f.is_dir) {
          if (/^(\.|__pycache__|node_modules|uploads$)/.test(f.name)) continue
          await walk(f.sub, depth + 1)
          continue
        }
        if (/\.(png|jpe?g|svg|gif|webp)$/i.test(f.name)) pics.push({ rel: f.rel_path, name: f.sub || f.name, mtime: f.mtime })
        if (found || !f.name.endsWith('.json') || (f.size ?? 0) < 200) continue
        if (seen.get(f.rel_path) === f.mtime) continue
        seen.set(f.rel_path, f.mtime)
        const v = await api.viz(f.rel_path).catch(() => null)
        if (v?.kind === 'field_series') { found = v as unknown as FieldSeries; foundAt = sub }
      }
    }
    timer = window.setTimeout(() => {
      walk('', 0).then(() => {
        if (dead) return
        if (found) { setField(found); setFieldAt(foundAt) }
        setPictures(pics.sort((a, b) => b.mtime - a.mtime).slice(0, 12))
      })
    }, settle ? 2500 : 0)
    return () => { dead = true; clearTimeout(timer) }
  }, [runId, settle])
  return { field, fieldAt, pictures }
}

export default function RunView({ id, config, groups }: {
  id: string; config: AppConfig | null; groups: ModelGroup[] | null
}) {
  const { session, events, link, notice, setNotice, send } = useRun(id)
  const [files, setFiles] = useState(false)
  const [reasoning, setReasoning] = useState(pref('openpaso.reasoning', 'shown') === 'shown')
  const [confirmStop, setConfirmStop] = useState(false)
  const [now, setNow] = useState(Date.now())
  const [busy, setBusy] = useState(false)
  const scroller = useRef<HTMLDivElement>(null)
  const pinned = useRef(true)

  const running = !!session?.running
  const outcome = running ? 'running' : (session?.outcome ?? 'running')
  const settle = events.filter((e) => e.type === 'tool_result' || e.type === 'done').length
  const { field, fieldAt, pictures } = useField(id, settle)
  // the folders solver steps wrote into: their files are shown under those steps
  const stepFolders = useMemo(() => {
    const out = new Set<string>()
    for (const e of events) {
      if (e.type !== 'tool_result' || !SOLVER_TOOLS.has(e.tool || '')) continue
      const f = stepFolder(String(e.result || ''), id)
      if (f) out.add(f)
    }
    return out
  }, [events, id])
  const inStep = (sub: string) => [...stepFolders].some((f) => sub === f || sub.startsWith(f + '/'))
  const leftover = {
    field: field && !inStep(fieldAt) ? field : null,
    pictures: pictures.filter((p) => !inStep(p.name.includes('/') ? p.name.slice(0, p.name.lastIndexOf('/')) : '')),
  }
  const [confirmDelete, setConfirmDelete] = useState(false)
  const model = findModel(groups, session?.model ?? null)

  useEffect(() => { if (!running) return; const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [running])
  useEffect(() => { setConfirmStop(false) }, [running, id])
  useEffect(() => { runsChanged() }, [running, outcome])

  // stay at the bottom while new work arrives, unless you scrolled up to read
  useEffect(() => {
    const el = scroller.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [events.length])

  const turnStart = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) if (events[i].type === 'turn_start' || events[i].type === 'user_msg') return events[i].t
    return undefined
  }, [events])
  const lastDone = useMemo(() => [...events].reverse().find((e) => e.type === 'done')?.t, [events])
  // a run that is not working stops its clock at its last sign of life, not at
  // the present moment: a run cut short two hours ago did not take two hours
  const lastSign = useMemo(() => [...events].reverse().find((e) => e.t)?.t, [events])
  const ended = lastDone ?? lastSign ?? now
  const seconds = turnStart ? ((running ? now : ended) - turnStart) / 1000 : 0
  const turnsSoFar = events.filter((e) => e.type === 'turn_start').length
  const steps = events.filter((e) => e.type === 'tool_call_pending').length
  const prompt = events.find((e) => e.type === 'user_msg')?.text
  // "Waiting for you" only while a Run this step button is on the screen: the
  // same events decide both, so the header cannot say it with nothing to press
  const waiting = running && (() => {
    const pend = new Set<string>()
    for (const e of events) {
      if (e.type === 'tool_call_pending') pend.add(e.call_id!)
      if (['tool_call_executing', 'tool_call_rejected', 'tool_result', 'tool_error'].includes(e.type)) pend.delete(e.call_id!)
      if (e.type === 'done') pend.clear()
    }
    return pend.size > 0
  })()
  const modelName = session?.model_detail || session?.claude_model
  // the running total arrives with each token count; the snapshot is from when the page opened
  const liveCost = useMemo(() => { for (let i = events.length - 1; i >= 0; i--) if (events[i].cost_usd_total != null) return events[i].cost_usd_total; return null }, [events])
  const cost = money(liveCost ?? session?.cost_usd)
  const firstStart = events.find((e) => e.type === 'turn_start' || e.type === 'user_msg')?.t
  const allSeconds = firstStart ? ((running ? now : ended) - firstStart) / 1000 : 0
  const kindWords = model?.kind === 'openrouter' ? 'hosted on OpenRouter' : model?.kind === 'claude-code' ? 'your Claude login'
    : model?.kind === 'local' ? 'on this machine' : ''

  if (link === 'missing') {
    return (
      <div className="h-full grid place-items-center">
        <div className="text-center">
          <p className="text-[20px] text-ink">This run does not exist.</p>
          <p className="mt-2 text-[15px] text-muted">It may have been deleted.</p>
          <button onClick={() => navigate({ view: 'home' })} className="mt-5 h-10 px-5 rounded-[8px] border line text-[15px] text-ink hover:bg-card">Start a new run</button>
        </div>
      </div>
    )
  }

  async function followUp(text: string, attach: File[]) {
    setBusy(true)
    try {
      let names: string[] = []
      if (attach.length) names = (await api.upload(id, attach)).saved.map((x) => x.name)
      if (running) {
        const note = names.length ? `${text}\n\n(I uploaded: ${names.map((n) => `uploads/${n}`).join(', ')})` : text
        return send({ type: 'steer', text: note })
      }
      return send({ type: 'prompt', text, attachments: names })
    } catch (e) {
      setNotice(String((e as Error).message || e)); return false
    } finally { setBusy(false) }
  }

  async function remove() {
    setConfirmDelete(false)
    try { await api.deleteRun(id); runsChanged(); navigate({ view: 'home' }) }
    catch (e) { setNotice(String((e as Error).message || e)) }
  }

  return (
    <div className="h-full flex flex-col relative">
      {/* ── what this run is, and its state ─────────────────────────────── */}
      <div className="border-b line px-10 pt-6 pb-4">
        <div className="max-w-[1040px] mx-auto">
          <div className="flex items-start gap-4">
            <h1 className="flex-1 min-w-0 text-[22px] font-semibold leading-[1.35] tracking-[-0.01em] text-ink line-clamp-2" title={prompt}>
              {prompt || (session ? 'New run' : 'Loading…')}
            </h1>
            <div className="flex items-center gap-2 shrink-0">
              <button onClick={() => setFiles(true)} className="h-9 px-3.5 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong">Files</button>
              <button onClick={() => setConfirmDelete(true)} disabled={running} title={running ? 'Stop the run before deleting it' : 'Delete this run from this machine'}
                      className="h-9 px-3.5 rounded-[8px] border line text-[14px] text-body hover:text-bad hover:border-bad/50 disabled:opacity-40 disabled:hover:text-body">Delete</button>
            </div>
          </div>

          <div className="mt-3 flex items-center gap-x-5 gap-y-2 flex-wrap text-[14px]" role="status">
            <span className={`flex items-center gap-2 text-[15px] font-medium ${STATE[outcome]?.tone}`} title={waiting ? undefined : STATE[outcome]?.means}>
              <Glyph outcome={outcome} /> {waiting ? 'Waiting for you' : STATE[outcome]?.word}
            </span>
            {turnStart && <span className="num text-muted" title="Time taken by the latest prompt or follow-up">{turnsSoFar > 1 ? `last turn ${clock(seconds)} · ${clock(allSeconds)} since the first prompt` : clock(seconds)}</span>}
            <span className="text-muted">{steps} step{steps === 1 ? '' : 's'}</span>
            {session && (
              <span className="text-body">
                {model?.label ?? session.model}{modelName ? <span className="num text-muted"> ({modelName})</span> : null}
                {kindWords && <span className="text-muted"> · {kindWords}</span>}
              </span>
            )}
            {cost
              ? <span className="num text-muted" title={model?.kind === 'claude-code' ? 'As reported by Claude Code' : "Estimated from token counts and OpenRouter's list price"}>{cost}</span>
              : session && model?.kind === 'openrouter' && steps > 0
                ? <span className="text-muted" title="No token counts or no price were reported for this run">cost unknown</span> : null}
            {running && (
              confirmStop ? (
                <span className="ml-auto flex items-center gap-2">
                  <span className="text-[14px] text-ink2">Stop this run and end everything it started?</span>
                  <button onClick={() => { send({ type: 'stop' }); setConfirmStop(false) }}
                          className="h-9 px-4 rounded-[8px] bg-bad text-white text-[14px] font-semibold">Stop</button>
                  <button onClick={() => setConfirmStop(false)} className="h-9 px-3 rounded-[8px] text-[14px] text-body hover:bg-card">Keep running</button>
                </span>
              ) : (
                <button onClick={() => setConfirmStop(true)}
                        className="ml-auto h-9 px-4 rounded-[8px] border line text-[14px] text-ink hover:border-bad/60 hover:text-bad">Stop</button>
              )
            )}
          </div>

          {confirmDelete && !running && (
            <div className="mt-3 flex items-center gap-2 flex-wrap" role="alertdialog" aria-label="Delete this run">
              <span className="text-[14px] text-ink2">Delete this run from this machine? Its record and every file in its folder are removed. This cannot be undone.</span>
              <button onClick={remove} className="h-9 px-4 rounded-[8px] bg-bad text-white text-[14px] font-semibold">Delete</button>
              <button onClick={() => setConfirmDelete(false)} className="h-9 px-3 rounded-[8px] text-[14px] text-body hover:bg-card">Keep it</button>
            </div>
          )}
          {link === 'lost' && (
            <p className="mt-3 text-[14px] text-bad">Connection to the server lost. Reconnecting… The run keeps working on the server.</p>
          )}
          {outcome === 'unfinished' && (
            <div className="mt-3 flex items-center gap-3 flex-wrap">
              <p className="text-[15px] text-body">
                {events.some((e) => e.type === 'error' && e.outcome === 'unfinished')
                  ? 'The server this run was working in was stopped, so the run stopped with it. Everything it had done is below.'
                  : 'This run has no recorded end and nothing is running for it now, so the server it was working in was stopped at some point.'}
              </p>
              <button onClick={() => void followUp(
                'Carry on from where you stopped. First say in one or two sentences what you had already done and what was left, then continue.', [])}
                      disabled={busy}
                      className="h-9 px-4 rounded-[8px] bg-coral text-on-coral text-[14px] font-semibold hover:bg-coral-h disabled:opacity-60">
                Carry on
              </button>
            </div>
          )}
          {notice && (
            <p className="mt-3 text-[14px] text-ink2 flex gap-3" role="alert">
              {notice}<button onClick={() => setNotice(null)} className="text-muted hover:text-ink underline underline-offset-4">Dismiss</button>
            </p>
          )}
        </div>
      </div>

      {/* ── the work ────────────────────────────────────────────────────── */}
      <div ref={scroller} className="flex-1 overflow-y-auto scroll"
           onScroll={(e) => { const el = e.currentTarget; pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80 }}>
        <div className="max-w-[1040px] mx-auto px-10 py-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-[13px] font-medium text-muted">What happened</span>
            <div role="radiogroup" aria-label="How much to show" className="ml-auto flex rounded-[8px] border line p-0.5">
              {([['everything', 'Everything'], ['steps', 'Steps only']] as const).map(([v, label]) => (
                <button key={v} role="radio" aria-checked={(v === 'everything') === reasoning}
                        title={v === 'steps'
                          ? 'The steps, their output and the reply. Leaves out the model\'s notes to itself and a critic\'s long verdicts.'
                          : 'Everything the run produced, in order.'}
                        onClick={() => { setReasoning(v === 'everything'); try { localStorage.setItem('openpaso.reasoning', v === 'everything' ? 'shown' : 'hidden') } catch { /* */ } }}
                        className={`h-7 px-3 rounded-[6px] text-[13px] ${(v === 'everything') === reasoning ? 'bg-card text-ink' : 'text-muted hover:text-ink'}`}>
                  {label}
                </button>
              ))}
            </div>
          </div>

          <Transcript events={events} live={running} showReasoning={reasoning} now={now} runId={id} modelKind={model ? model.kind : session?.model === 'mock' ? 'test' : undefined}
                      onEndStep={model?.kind === 'claude-code' ? undefined : (cid) => send({ type: 'end_step', call_id: cid })}
                      onDecide={(cid, ok) => send(ok ? { type: 'approve', call_id: cid } : { type: 'reject', call_id: cid, reason: 'skipped by the user' })} />

          {/* A solver step shows what it wrote under itself. What no solver step wrote (a
              picture a shell command drew, say) is shown here, after the run, so nothing the
              run made is out of sight and nothing is shown as the result of a step it is not. */}
          {(leftover.field || leftover.pictures.length > 0) && (
            <section className="mt-8" aria-label="Other files the run made">
              <div className="text-[13px] font-medium text-muted mb-2">Pictures in the run's folder that no solver step wrote</div>
              {leftover.field && <div className="mb-3"><Stage series={leftover.field} /></div>}
              <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
                {leftover.pictures.map((p) => (
                  <figure key={p.rel} className="bg-soft border line rounded-[8px] p-3">
                    <img src={fileUrl(p.rel)} alt={p.name} loading="lazy" className="w-full rounded-[6px] bg-white" />
                    <figcaption className="num mt-2 text-[13px] text-muted break-all">{p.name}</figcaption>
                  </figure>
                ))}
              </div>
            </section>
          )}
        </div>
      </div>

      {/* ── say something to it ─────────────────────────────────────────── */}
      <div className="border-t line px-10 py-4">
        <div className="max-w-[1040px] mx-auto">
          <Composer
            placeholder={!running
              ? 'Ask a follow-up about this run, or tell openPASO what to change and run again'
              : model?.kind === 'claude-code'
                ? 'Send a correction. Claude Code cannot be interrupted, so it is sent the moment this turn ends.'
                : 'Send a correction. openPASO reads it as soon as the current step finishes.'}
            submitLabel={running ? 'Send correction' : 'Send'}
            onSubmit={followUp} busy={busy} draftKey={`run.${id}`}
            left={session && (
              <ModePicker modes={config?.modes} value={session.mode} planAllowed={session.model !== 'claude-code'}
                          onChange={(m) => send({ type: 'set_mode', mode: m })} />
            )} />
        </div>
      </div>

      <FilesDrawer runId={id} open={files} onClose={() => setFiles(false)} refreshKey={settle} />
    </div>
  )
}

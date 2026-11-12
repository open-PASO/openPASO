import type { Ev } from '../types'
import { SOLVER_TOOLS, stepVerdict } from './Transcript'

/* Where a run is, in six plain stages, read from the steps it took.

   The transcript says everything that happened; this says what kind of work it was, so a
   person can see at a glance whether the agent is still finding its way, building inputs,
   being reviewed, solving, or done. A stage is reached only by a step that ran; nothing is
   inferred from what the model wrote. */

export type Stage = 'understand' | 'build' | 'review' | 'solve' | 'check' | 'report'

const ORDER: [Stage, string, string][] = [
  ['understand', 'Understand', 'Reads the task and what openPASO knows: solvers, methods, examples, input data'],
  ['build', 'Build', 'Writes and tests the solver inputs and scripts'],
  ['review', 'Review', 'A critic, the same model given only the work, looks for what is wrong'],
  ['solve', 'Solve', 'An openPASO solver or coupling runs'],
  ['check', 'Check', "openPASO's own checks of the solver result"],
  ['report', 'Report', 'The reply with the answer'],
]

const SOLVER = new Set(['run_simulation', 'run_with_generator', 'coupled_solve', 'couple', 'couple_levels',
                        'couple_precice'])
const CHECKS = new Set(['verify_mesh_independence', 'verify_pde_consistency', 'verify_interface_flux',
                        'audit_results', 'check_input'])
const LOOKUP = new Set(['discover', 'knowledge', 'examples', 'web_search', 'WebSearch', 'prepare_simulation',
                        'material_handoff', 'list_solvers', 'get_coupling_knowledge'])

function stageOf(e: Ev): Stage | null {
  const role = typeof e.args?.role === 'string' ? e.args.role : ''
  // Claude Code runs its critic with its own sub-agent tool, as Transcript reads it
  const claudeCritic = (e.tool === 'Agent' || e.tool === 'Task') && /critic|review/i.test(JSON.stringify(e.args || {}))
  if (e.agent === 'critic' || e.tool === 'submit_critic_review' || claudeCritic ||
      (e.tool === 'spawn_subagent' && role === 'critic'))
    return 'review'
  if (e.tool === 'spawn_subagent' && role === 'verifier') return 'check'
  if (SOLVER.has(e.tool || '')) return 'solve'
  if (CHECKS.has(e.tool || '')) return 'check'
  if (LOOKUP.has(e.tool || '')) return 'understand'
  return e.tool ? 'build' : null
}

const RANK = { failed: 1, unverified: 2, verified: 3 } as const
type Verdict = keyof typeof RANK

export function progress(events: Ev[], running: boolean) {
  const count: Record<Stage, number> = { understand: 0, build: 0, review: 0, solve: 0, check: 0, report: 0 }
  let current: Stage | null = null
  // the latest turn's best solver evidence, ranked as the transcript ranks a turn's outcome
  let verdict: Verdict | null = null
  let turnBest: Verdict | null = null
  const pending = new Map<string, Ev>()       // a step's tool, arguments and agent, from before it ran
  const ran = new Set<string>()
  const skipped = new Set<string>()
  let saidSinceStep = false
  const step = (id: string) => {
    const p = pending.get(id)
    if (!p || ran.has(id) || skipped.has(id)) return
    ran.add(id)
    const s = stageOf(p)
    if (s) { count[s] += 1; current = s }
  }
  for (const e of events) {
    const id = e.call_id || ''
    if (e.type === 'turn_start') {
      current = null; turnBest = null; saidSinceStep = false
    } else if (e.type === 'tool_call_pending') {
      pending.set(id, e); saidSinceStep = false
    } else if (e.type === 'tool_call_executing') {
      step(id)                                // counted when it runs, not when it is proposed
    } else if (e.type === 'tool_call_rejected') {
      skipped.add(id)
    } else if (e.type === 'tool_result' || e.type === 'tool_error') {
      if (e.type === 'tool_result' && /^\[rejected by user/m.test(String(e.result || ''))) { skipped.add(id); continue }
      step(id)                                // a record from before steps reported running
      const tool = pending.get(id)?.tool || e.tool || ''
      if (SOLVER_TOOLS.has(tool)) {
        const v: Verdict = e.type === 'tool_error' ? 'failed' : stepVerdict(e)
        if (!turnBest || RANK[v] > RANK[turnBest]) turnBest = v
        verdict = turnBest
        if (!CHECKS.has(tool)) count.check += 1     // a mesh study's run already counted as a check
        current = 'check'
      }
    } else if (e.type === 'agent_msg') {
      if ((e.text || '').trim()) saidSinceStep = true
    } else if (e.type === 'done') {
      // a turn that failed or was stopped has no answer, and neither has one whose last act was a step
      const answered = !['failed', 'interrupted', 'unfinished'].includes(String(e.outcome || '')) && saidSinceStep
      if (answered) count.report += 1
      current = answered ? 'report' : null
    }
  }
  return { count, current: running ? current : null, verdict }
}

export default function Progress({ events, running }: { events: Ev[]; running: boolean }) {
  const { count, current, verdict } = progress(events, running)
  if (!Object.values(count).some(Boolean)) return null
  return (
    <ol className="mt-4 grid grid-cols-6 gap-1.5" aria-label="Where the run is">
      {ORDER.map(([key, word, means]) => {
        const n = count[key]
        const now = current === key
        const tone = key === 'check' && verdict && verdict !== 'verified' ? 'coral' : now ? 'coral' : n ? 'ink' : 'faint'
        const note = key === 'check' && verdict
          ? verdict === 'verified' ? 'verified' : verdict === 'failed' ? 'failed' : 'not verified'
          : n ? `${n} step${n === 1 ? '' : 's'}` : ''
        const glyph = key !== 'check' || !verdict ? '✓' : verdict === 'verified' ? '✓' : verdict === 'failed' ? '✕' : '△'
        return (
          <li key={key} title={means} aria-current={now ? 'step' : undefined}
              className={`rounded-[8px] border px-3 py-2 ${now ? 'border-coral/60 bg-coral/[0.05]' : n ? 'line bg-soft' : 'line'}`}>
            <div className={`text-[13px] font-semibold ${tone === 'coral' ? 'text-coral' : tone === 'ink' ? 'text-ink2' : 'text-faint'}`}>
              {n && !now ? <span aria-hidden className="num mr-1.5">{glyph}</span> : null}
              {word}
            </div>
            <div className={`num text-[12px] ${n ? 'text-muted' : 'text-faint'}`}>{now && running ? 'now' : note || '—'}</div>
          </li>
        )
      })}
    </ol>
  )
}

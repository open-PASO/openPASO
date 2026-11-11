import type { Ev } from '../types'

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
  if (e.agent === 'critic' || e.tool === 'submit_critic_review' || (e.tool === 'spawn_subagent' && role === 'critic'))
    return 'review'
  if (e.tool === 'spawn_subagent' && role === 'verifier') return 'check'
  if (SOLVER.has(e.tool || '')) return 'solve'
  if (CHECKS.has(e.tool || '')) return 'check'
  if (LOOKUP.has(e.tool || '')) return 'understand'
  return e.tool ? 'build' : null
}

export function progress(events: Ev[], running: boolean) {
  const count: Record<Stage, number> = { understand: 0, build: 0, review: 0, solve: 0, check: 0, report: 0 }
  let current: Stage | null = null
  let verdict: 'verified' | 'unverified' | 'failed' | null = null
  for (const e of events) {
    if (e.type === 'tool_call_pending') {
      const s = stageOf(e)
      if (s) { count[s] += 1; current = s }
    } else if (e.type === 'tool_result' && SOLVER.has(e.tool || '') && e.verdict) {
      verdict = e.verdict === 'verified' ? 'verified' : e.verdict === 'failed' ? 'failed' : 'unverified'
      count.check += 1
      current = 'check'
    } else if (e.type === 'done') {
      count.report += 1
      current = 'report'
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

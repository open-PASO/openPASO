import { useMemo, useState } from 'react'
import { clock, tidy } from '../api'
import type { Ev } from '../types'
import Markdown from './Markdown'
import StepOutputs, { stepFolder } from './StepOutputs'

/* The run, in the order it happened.

   Everything here comes from the run's own event log. Every statement about how
   a step or a turn ended follows one rule, the same one the server applies
   (webui/outcome.py): a solver tool reports its own verdict, and only a result
   it reports as completed and trustworthy is a finished result. */

// the same list the server uses; a shell command that exits zero is not a solver
const SOLVER_TOOLS = new Set(['run_simulation', 'run_with_generator', 'coupled_solve', 'couple',
                              'couple_levels', 'couple_precice', 'verify_mesh_independence'])
const SHELL_TOOLS = new Set(['run_bash', 'Bash'])

/** The text inside MCP's content blocks, escapes turned back into characters. */
function readable(raw: string): string {
  // Claude Code hands an MCP tool's reply back as {"result": "<the reply>"}: read the
  // reply, not the wrapper (read as it was, a finished run's verdict came out "failed")
  const trimmed = raw.trim()
  if (trimmed.startsWith('{"result"')) {
    try {
      const w = JSON.parse(trimmed) as Record<string, unknown>
      if (typeof w.result === 'string' && Object.keys(w).length === 1) return readable(w.result)
    } catch { /* a shortened copy is not whole JSON: its escapes are read below */ }
  }
  // MCP hands a result over as a Python repr of text blocks, whose escapes are
  // the wrapper's. Undo those only where the wrapper really is: a result that
  // is plain text keeps its backslashes, so a command or a formula containing
  // \nabla is not turned into a line break and lost.
  const blocks = [...raw.matchAll(/'text':\s*(['"])([\s\S]*?)\1\s*(?:,\s*'|})/g)].map((m) => m[2])
  if (!blocks.length) return tidy(raw)
  const t = blocks.join('\n')
    .replace(/\\u([0-9a-fA-F]{4})/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
    .replace(/\\n/g, '\n').replace(/\\t/g, '  ').replace(/\\'/g, "'").replace(/\\"/g, '"')
    .replace(/\\\\/g, '\\')
  return tidy(t)
}

/** 'verified', 'unverified' or 'failed' for one solver tool result, with the
    reason in plain words. Mirrors classify_solver_result in webui/outcome.py. */
/** The verdict a report states about itself, if it states one. */
function verdictOf(node: Record<string, unknown>): 'verified' | 'unverified' | 'failed' | null {
  const status = String(node.status ?? '').toLowerCase()
  const trusted = node.trustworthy_result
  if (status) {
    if (!status.startsWith('completed')) return 'failed'
    return status === 'completed' && trusted === true ? 'verified' : 'unverified'
  }
  if (trusted === true) return 'verified'
  if (trusted === false || 'converged' in node || 'all_levels_converged' in node) {
    return node.error ? 'failed' : 'unverified'
  }
  return node.error ? 'failed' : null
}

function solverVerdict(raw: string): { verdict: 'verified' | 'unverified' | 'failed'; reason: string } {
  const t = readable(raw)
  // The report's own top-level verdict decides, the same rule the server
  // applies: a ladder of couplings answers for the ladder, and one verified
  // level inside a ladder that failed does not make the ladder verified.
  const start = t.indexOf('{')
  if (start >= 0) {
    try {
      const doc = JSON.parse(t.slice(start, t.lastIndexOf('}') + 1)) as Record<string, unknown>
      const top = verdictOf(doc)
      if (top === 'verified') return { verdict: 'verified', reason: '' }
      if (top) {
        const err = typeof doc.error === 'string' ? doc.error : ''
        const named = typeof doc.what_the_error_is === 'string' ? doc.what_the_error_is : ''
        const note = typeof doc.verification === 'string' ? doc.verification : ''
        // a failed run is explained by what broke, not by the verification sentence
        // every failure shares (the server's rule: verdict_reason in webui/outcome.py)
        const why = top === 'failed'
          ? (err ? (troubleOf(err) || err) + (named ? ` -- ${named}` : '') : '') || note
          : note || err
        return { verdict: top, reason: cut(why || (top === 'failed' ? 'the solver reported a failure'
                 : 'ran, but openPASO did not verify the result'), 160) }
      }
    } catch { /* not a whole JSON document; read it as text below */ }
  }
  // Two shapes, the same two the server reads (webui/outcome.py): a run reports
  // "status" plus the verification gate's "trustworthy_result"; a coupling
  // reports no status at all, only the gate's verdict beside "converged".
  // The same rule the server applies to a record it cannot parse: these
  // matches may belong to different objects, so a mixture is never verified —
  // pairing a healthy participant with a failed report once read as "Finished".
  // the quotes may be escaped: a shortened copy of a reply wrapped as {"result": "..."}
  const flags = [...t.matchAll(/\\?"trustworthy_result\\?"\s*:\s*(true|false)/g)].map((m) => m[1])
  const trusted = flags.length > 0 && flags.every((f) => f === 'true')
  const untrusted = flags.includes('false')
  const statuses = [...t.matchAll(/\\?"status\\?"\s*:\s*\\?"([A-Za-z_]+)\\?"/g)].map((m) => m[1].toLowerCase())
  // one status in the whole text can only be this report's own; several, and
  // there is no way to tell whose, so none of them speaks for the whole
  const status = statuses.length === 1 ? statuses[0] : ''
  const err = t.match(/"(?:error|message)"\s*:\s*"([^"\n]{1,140})/)
  const unverified = { verdict: 'unverified' as const, reason: 'ran, but openPASO did not verify the result' }
  // The status decides first, exactly as the server does. Checking the
  // verification flag first made "completed_with_warnings" with a trusted flag
  // read as verified here while the header and the record said otherwise.
  if (status) {
    if (!status.startsWith('completed')) {
      return { verdict: 'failed', reason: err ? err[1] : `the solver reported “${status}”` }
    }
    return status === 'completed' && trusted ? { verdict: 'verified', reason: '' } : unverified
  }
  if (trusted) return { verdict: 'verified', reason: '' }
  if (untrusted || /"(?:converged|all_levels_converged)"\s*:/.test(t)) {
    return err ? { verdict: 'failed', reason: err[1] } : unverified
  }
  // the legacy coupled_solve answers in prose with openPASO's verification note
  if (/\[openPASO verification:/i.test(t) && !/^(?:Backend not found|Unknown problem|Unknown solver|Error|Traceback)/im.test(t)) {
    return unverified
  }
  // what broke, not the banner above it: a solver that ended in a traceback
  // was being reported as "Traceback (most recent call last):"
  const line = troubleOf(t) || (t.split('\n').map((l) => l.trim()).find(Boolean) || 'no result')
  return { verdict: 'failed', reason: cut(line, 160) }
}

/** A line shortened for the one-line summary says it was shortened: a cut with
    no mark reads as the whole sentence, and someone answers from what is not
    there. The full output is always one click away under the step. */
const cut = (text: string, n: number) => (text.length > n ? text.slice(0, n) + '…' : text)


/** The failure an ordinary tool result reports, if it reports one. */
export function troubleOf(raw: string): string | null {
  // read it as a person would first: the failure line was being shown as the
  // Python repr the tool layer wraps it in
  const t = readable(raw)
  const timeout = t.match(/\[timeout after [^\]]+\]/)
  if (timeout) return timeout[0].replace(/^\[|\]$/g, '')
  if (/Error executing tool|validation error for/i.test(t)) {
    const line = t.split('\n').find((l) => /error/i.test(l)) || 'the tool refused its input'
    return cut(line.trim(), 140)
  }
  const exc = t.match(/^\s*([A-Za-z_][\w.]*(?:Error|Exception|Fault)): ?(.*)$/m)
  if (exc) return cut(`${exc[1]}: ${exc[2]}`.trim(), 140)
  if (/Traceback \(most recent call last\)/.test(t)) {
    // the line that says what went wrong is the LAST one, not the banner
    const raised = [...t.matchAll(/^\s*([A-Za-z_][\w.]*(?:Error|Exception|Fault|Warning)): ?(.*)$/gm)]
    const last = raised[raised.length - 1]
    return last ? cut(`${last[1]}: ${last[2]}`.trim(), 160) : 'the command ended in a traceback'
  }
  const nonzero = t.match(/non-zero exit status (\d+)|exit(?: code|ed with)? (\d+)/i)
  if (nonzero && (nonzero[1] || nonzero[2]) !== '0') return `exit code ${nonzero[1] || nonzero[2]}`
  if (/could not search|returned nothing after three attempts/i.test(t))
    return 'the search provider refused; this is not evidence the web has nothing'
  if (/^\[no results\]/m.test(t)) return 'no results'
  if (/^\[rejected by user/m.test(t)) return 'you skipped this step'
  return null
}

function argLine(tool: string, a?: Record<string, unknown>): string {
  if (!a) return ''
  const s = (k: string) => (typeof a[k] === 'string' ? (a[k] as string) : '')
  let out: string
  switch (tool) {
    case 'run_bash': case 'Bash': out = s('command'); break
    case 'write_file': case 'Write': out = `${s('path') || s('file_path')}  (${(s('content')).length.toLocaleString()} characters)`; break
    case 'read_file': case 'Read': out = s('path') || s('file_path'); break
    case 'discover': case 'knowledge': case 'examples': case 'web_search': case 'WebSearch': out = s('query'); break
    case 'prepare_simulation': out = [s('solver'), s('physics')].filter(Boolean).join(' · '); break
    default: {
      // critic_approved and critic_token are what the model claimed, and openPASO does not take
      // them as evidence: shown in the step line they read as "the critic approved" beside a
      // verdict that says no review matches
      const k = Object.keys(a).filter((x) => typeof a[x] !== 'object'
                                              && x !== 'critic_approved' && x !== 'critic_token')
      out = k.map((x) => `${x}=${String(a[x])}`).join('  ')
    }
  }
  return tidy(out)
}

/** What a step does, in words, for the tools openPASO's runs use. A tool not
    listed shows its name only; nothing is guessed. */
const DOES: Record<string, string> = {
  run_bash: 'Run a shell command, starting in the run folder',
  Bash: 'Run a shell command',
  write_file: 'Write a file', Write: 'Write a file',
  read_file: 'Read a file', Read: 'Read a file',
  web_search: 'Search the web', WebSearch: 'Search the web',
  discover: 'Look up which solvers and kinds of problem openPASO knows',
  knowledge: "Read openPASO's notes on a solver or method",
  examples: 'Look up a worked input example',
  prepare_simulation: 'Prepare the input for a solver',
  run_simulation: 'Run a solver',
  run_with_generator: 'Generate a solver input and run it',
  coupled_solve: 'Run two solvers coupled together',
  couple: 'Run two solvers coupled together',
  couple_precice: 'Run two solvers coupled through preCICE',
  generate_mesh: 'Make a mesh',
  visualize: 'Make a picture of a result',
  transfer_field: 'Move a field from one mesh to another',
  setup_backend: 'Install or configure a solver',
  developer: "Read or change a solver's source code",
  verify_pde_consistency: 'Check the equations and set-up for consistency',
  audit_results: 'Check a result file for problems',
  submit_critic_review: "Record the critic's review",
}

const ADVICE: [RegExp, string][] = [
  [/no openrouter key/i, 'Add your OpenRouter key: put the line OPENROUTER_API_KEY=... in a .env file in the openPASO folder, then send a follow-up.'],
  [/claude code is not on path/i, 'Install Claude Code, or start a new run with another model.'],
  [/claude code exited/i, 'Check that Claude Code is signed in (run `claude` once in a terminal), then send a follow-up.'],
  [/402|insufficient credits|payment required/i, 'Your OpenRouter account is out of credit. Top it up, or start a new run with another model.'],
  [/401|unauthori[sz]ed|invalid api key/i, 'The OpenRouter key was rejected. Check OPENROUTER_API_KEY.'],
  [/429|rate limit/i, 'The model provider is rate-limiting. Wait a minute and send a follow-up.'],
  [/connection refused|connecterror|apiconnectionerror|connection error/i, 'The model server could not be reached. For a local model, start its server; for a hosted one, check the network.'],
  [/recursion ?limit|steps it is allowed/i, 'Send a follow-up asking for a smaller piece of the problem.'],
]

type CallState = 'waiting' | 'running' | 'done' | 'unverified' | 'failed' | 'skipped' | 'abandoned'
type Call = {
  kind: 'call'; id: string; tool: string; agent: string; args?: Record<string, unknown>
  state: CallState; detail: string; result?: string; t0?: number; t1?: number; ending?: boolean
  /** a review of the work, filed by the one who did the work: no critic came back since the
      last filed review, or the critic that did said the opposite */
  selfReview?: 'no critic' | 'overrules the critic'
}
type Tally = { tools: number; shell: number; verified: boolean; unverified: boolean
               solverFailed: string | null; said: boolean; why: string }
type Entry =
  | { kind: 'turn' }
  | { kind: 'you'; text: string; files: string[] }
  | { kind: 'thought'; text: string; final: boolean; outcome?: string; typed?: string[] }
  | Call
  | { kind: 'aside'; id: string; role: string; task: string; returned: boolean }
  | { kind: 'verdict'; role: string; text: string; reached: boolean }
  | { kind: 'steer'; id: string; text: string; state: string }
  | { kind: 'upload'; name: string; bytes: number }
  | { kind: 'mode'; mode: string }
  | { kind: 'end'; outcome: string; tally: Tally; message?: string; traceback?: string; seconds?: number; left?: number }

/** APPROVED or REJECTED, as a review states it: its VERDICT line first, else the first line
    that opens with one of the two words. Anything else is neither. */
function reviewWord(text: string): 'approved' | 'rejected' | null {
  const t = text || ''
  const m = t.match(/VERDICT\W{0,4}(APPROVED|REJECTED)/i) || t.match(/^\W{0,4}(APPROVED|REJECTED)\b/im)
  return m ? (m[1].toUpperCase() === 'APPROVED' ? 'approved' : 'rejected') : null
}

/** Steps a reply wrote out as text ("[step] tool {...}" then "→ result"). None of them ran.
    The server's rule (webui/outcome.py typed_steps), mirrored for records it did not stamp. */
export function typedSteps(text: string): string[] {
  return Array.from((text || '').matchAll(/^\[(?:[\w/-]+ )?step\] ([\w.-]+)/gm), m => m[1])
}

/** Was a filed review put on record? Refused filings (by openPASO, or by this interface, which
    lets only the critic file one) recorded nothing. */
function reviewReply(raw: string): { accepted: boolean; error: string } {
  const t = readable(raw || '')
  try {
    const j = JSON.parse(t)
    return { accepted: j?.accepted === true, error: String(j?.error || '') }
  } catch {
    return { accepted: /"accepted":\s*true/.test(t), error: '' }
  }
}

/** Did a sub-agent actually reach a conclusion, or only stop? */
function concluded(text: string): boolean {
  const t = (text || '').trim()
  if (!t) return false
  return !/^\[sub-agent error|need more steps/i.test(t)
      && !/returned no text|returned nothing/i.test(t.slice(0, 120))
}

const tally0 = (): Tally => ({ tools: 0, shell: 0, verified: false, unverified: false,
                               solverFailed: null, said: false, why: '' })

function build(events: Ev[], live: boolean): Entry[] {
  const out: Entry[] = []
  const calls = new Map<string, Call>()
  const asides = new Map<string, Extract<Entry, { kind: 'aside' }>>()
  const steers = new Map<string, Extract<Entry, { kind: 'steer' }>>()
  let stream = ''
  let turns = 0, turnT = 0
  let tally = tally0()
  // did a critic reach a verdict since the last review was filed, and which one
  let criticSpoke = false
  let criticSaid: 'approved' | 'rejected' | null = null
  let lastError: Ev | null = null
  const attachedLater = new Set<string>()
  events.forEach((e) => { if (e.type === 'user_msg') (e.attachments || []).forEach((a) => attachedLater.add(a)) })

  for (const e of events) {
    switch (e.type) {
      case 'turn_start':
        turns += 1; turnT = e.t || 0; lastError = null; tally = tally0(); criticSpoke = false; criticSaid = null
        if (turns > 1) out.push({ kind: 'turn' })
        break
      case 'user_msg':
        if (turns === 0) { turns = 1; turnT = e.t || 0 }
        out.push({ kind: 'you', text: e.text || '', files: e.attachments || [] })
        break
      case 'file_uploaded':
        if (!attachedLater.has(e.name || '')) out.push({ kind: 'upload', name: e.name || '', bytes: e.bytes || 0 })
        break
      case 'agent_chunk':
        stream += e.text || ''
        break
      case 'agent_msg': {
        stream = ''
        const t = (e.text || '').trim()
        if (t) { out.push({ kind: 'thought', text: t, final: false }); tally.said = true }
        break
      }
      case 'tool_call_pending': {
        const c: Call = { kind: 'call', id: e.call_id || '', tool: e.tool || '', agent: e.agent || 'main',
                          args: e.args, state: 'waiting', detail: '' }
        calls.set(c.id, c); out.push(c)
        break
      }
      case 'tool_call_executing': {
        const c = calls.get(e.call_id || ''); if (c) { c.state = 'running'; c.t0 = e.t }
        break
      }
      case 'step_ending': {
        const c = calls.get(e.call_id || ''); if (c) c.ending = true
        break
      }
      case 'tool_result': {
        const c = calls.get(e.call_id || '')
        if (!c) break
        // Claude Code runs its critic with its own sub-agent tool (Agent / Task), not with
        // openPASO's spawn_subagent: a conclusion that came back that way is a critic's too.
        // Missing this, a Claude Code run whose critic reviewed twice was told it had
        // reviewed its own work.
        if ((c.tool === 'Agent' || c.tool === 'Task') && /critic|review/i.test(JSON.stringify(c.args || {}))
            && concluded(readable(e.result || ''))) {
          criticSpoke = true
          criticSaid = reviewWord(readable(e.result || ''))
        }
        if (c.tool === 'submit_critic_review' && c.agent === 'main' && reviewReply(e.result || '').accepted) {
          // Measured 2026-09-29: the critic came back REJECTED, and the model then filed
          // "VERDICT: APPROVED" five times, rewriting its script in between, with no critic
          // run after the first. A filed review is the critic's only when a critic came back
          // since the last one was filed, and only if it does not say the opposite. (Since
          // 2026-09-30 this interface refuses the main agent's filing; this is for records
          // written before, and for Claude Code runs, whose calls it does not carry.)
          const filed = reviewWord(String(c.args?.findings ?? ''))
          if (!criticSpoke) c.selfReview = 'no critic'
          else if (filed === 'approved' && criticSaid === 'rejected') c.selfReview = 'overrules the critic'
          criticSpoke = false
        }
        c.result = e.result || ''; c.t1 = e.t
        if (c.ending || c.result.startsWith('[The user ended this step')) {
          c.state = 'abandoned'; c.detail = 'you ended this step'; tally.tools += 1
          break
        }
        tally.tools += 1
        if (SHELL_TOOLS.has(c.tool)) tally.shell += 1
        if (SOLVER_TOOLS.has(c.tool)) {
          // what the server decided, where it said so; its own reading only
          // for a record written before the server stamped its verdict
          const stamped = e.verdict as 'verified' | 'unverified' | 'failed' | undefined
          const own = solverVerdict(c.result)
          const said = typeof e.why === 'string' ? e.why.trim() : ''
          const v = stamped
            ? { verdict: stamped, reason: said || (stamped === own.verdict ? own.reason
                : stamped === 'verified' ? '' : own.reason) }
            : own
          if (said) tally.why = said
          c.detail = v.reason
          if (v.verdict === 'verified') { c.state = 'done'; tally.verified = true }
          else if (v.verdict === 'unverified') { c.state = 'unverified'; tally.unverified = true }
          else { c.state = 'failed'; tally.solverFailed = tally.solverFailed ?? v.reason }
        } else if (c.tool === 'submit_critic_review' && !reviewReply(c.result).accepted) {
          c.state = 'failed'
          c.detail = cut(tidy('not recorded: ' + (reviewReply(c.result).error || 'the review was refused')), 160)
        } else {
          const bad = troubleOf(c.result)
          c.state = bad === 'you skipped this step' ? 'skipped' : bad ? 'failed' : 'done'
          c.detail = bad || ''
        }
        break
      }
      case 'tool_error': {
        const c = calls.get(e.call_id || '')
        if (c) {
          c.state = 'failed'
          c.detail = cut(tidy(e.error || e.message || 'failed'), 160)
          c.t1 = e.t
          tally.tools += 1
          // a solver tool that raised has been called and produced nothing;
          // without this the closing line said no solver ran at all
          if (SOLVER_TOOLS.has(c.tool)) tally.solverFailed = tally.solverFailed ?? c.detail
        }
        break
      }
      case 'tool_call_rejected': {
        const c = calls.get(e.call_id || '')
        if (c) { c.state = 'skipped'; c.detail = 'you skipped this step'; c.t1 = e.t }
        break
      }
      case 'subagent_spawned': {
        const a = { kind: 'aside' as const, id: e.sa_id || '', role: e.role || 'sub-agent', task: e.task || '', returned: false }
        asides.set(a.id, a); out.push(a)
        break
      }
      case 'subagent_returned': {
        // a real conclusion, from a critic: not the sentinel that says it
        // returned nothing, not an error, and not a researcher's summary
        if (asides.get(e.sa_id || '')?.role === 'critic' && concluded(e.result || '')) {
          criticSpoke = true
          criticSaid = reviewWord(e.result || '')
        }
        // shown where it came back, after the sub-agent's own steps
        const a = asides.get(e.sa_id || '')
        const text = (e.result || '').trim()
        const reached = concluded(text)
        if (a) a.returned = true
        // the sub-agent's last message IS what it returns: shown once, as its conclusion
        // (it was printed twice, as a message and again under "What the critic concluded")
        for (let j = out.length - 1; j >= 0; j--) {
          const x = out[j]
          if (x.kind === 'thought' && x.text.trim() === text) { out.splice(j, 1); break }
          if (x.kind === 'aside' || x.kind === 'turn' || x.kind === 'you') break
        }
        out.push({ kind: 'verdict', role: a?.role || 'sub-agent', text, reached })
        break
      }
      case 'user_steer': {
        const s = { kind: 'steer' as const, id: e.id || '', text: e.text || '', state: e.state || 'queued' }
        steers.set(s.id, s); out.push(s)
        break
      }
      case 'steer_state': {
        const s = steers.get(e.id || ''); if (s) s.state = e.state || s.state
        break
      }
      case 'mode_changed':
        out.push({ kind: 'mode', mode: e.mode || '' })
        break
      case 'error':
        lastError = e
        break
      case 'done': {
        // the server's rule: an error decides the turn, and an error without an
        // outcome of its own is a failure; a clean end is judged by the solver
        // evidence, never by what an old record's "done" claimed
        let outcome: string
        if (lastError) outcome = lastError.outcome || 'failed'
        else if (e.outcome === 'failed' || e.outcome === 'interrupted') outcome = e.outcome
        else if (e.by === 'server' && e.outcome) outcome = e.outcome   // decided once, on the server
        else outcome = tally.verified ? 'completed' : tally.unverified ? 'unverified' : 'no_result'
        for (const c of calls.values()) {
          if (c.state === 'running') {
            c.state = 'abandoned'
            c.detail = outcome === 'interrupted' ? 'stopped while running, result unknown'
              : outcome === 'failed' ? 'cut off when the run failed, result unknown' : 'did not finish, result unknown'
          }
          if (c.state === 'waiting') { c.state = 'abandoned'; c.detail = 'never ran' }
        }
        // the model's last words in a turn that did not fail are its reply
        if (outcome !== 'failed') {
          for (let i = out.length - 1; i >= 0; i--) {
            const x = out[i]
            if (x.kind === 'call' || x.kind === 'turn' || x.kind === 'you' || x.kind === 'verdict') break
            if (x.kind === 'thought') {
              x.final = true; x.outcome = outcome
              x.typed = e.typed_steps && e.typed_steps.length ? e.typed_steps : typedSteps(x.text)
              break
            }
          }
        }
        out.push({ kind: 'end', outcome, tally: { ...tally },
                   message: lastError?.message, traceback: lastError?.traceback,
                   seconds: turnT && e.t ? (e.t - turnT) / 1000 : undefined, left: lastError?.processes_left })
        lastError = null
        break
      }
    }
  }
  if (live && stream.trim()) out.push({ kind: 'thought', text: stream.trim(), final: false })
  if (!live) {
    // the run is not working, so nothing in it is. A run cut off by the server
    // going down has no closing event, and its steps used to keep a running
    // clock for ever, which said work was happening when none was.
    for (const c of calls.values()) {
      if (c.state === 'running') { c.state = 'abandoned'; c.detail = c.detail || 'the run stopped here; no result from this step' }
      if (c.state === 'waiting') { c.state = 'abandoned'; c.detail = 'never ran' }
    }
  }
  return out
}

function endText(x: Extract<Entry, { kind: 'end' }>): string {
  const after = x.seconds ? ` after ${clock(x.seconds)}` : ''
  const t = x.tally
  switch (x.outcome) {
    case 'completed':
      return `Finished${x.seconds ? ` in ${clock(x.seconds)}` : ''}. An openPASO solver ran and openPASO verified its result.`
    case 'unverified':
      // the solver's own account of what it checked and what it found: a ladder
      // whose levels are each sound but whose mesh never refined needs that
      // distinction, not a sentence that fits every unverified result alike
      return t.why
        ? `Ended${after}. A solver ran, but openPASO did not verify its result. What it reported:`
        : `Ended${after}. A solver ran, but openPASO did not verify its result. Check the numbers before you rely on them.`
    case 'failed':
      return `Failed${after}.`
    case 'interrupted':
      return x.message || 'Stopped.'
    case 'unfinished':
      // the server wrote why when it went down; do not replace it with a guess
      return x.message || `Stopped without finishing${after}. The server this run was working in went down.`
    default:
      if (t.solverFailed) return `Ended${after}. A solver was called but computed nothing: ${t.solverFailed}.`
      if (t.tools === 0 && !t.said) return `Ended${after}. Nothing happened in this turn: no tools were used and the model said nothing.`
      if (t.tools === 0) return `Ended${after}. No tools were used. The reply above is the model's own text, not a computation.`
      if (t.shell > 0) return `Ended${after}. openPASO's solvers were not used. Any numbers above come from commands and scripts the model ran itself (${t.shell} shell step${t.shell === 1 ? '' : 's'}). Check them before you rely on them.`
      return `Ended${after}. No solver ran in this turn.`
  }
}

const GLYPH: Record<CallState, string> = {
  waiting: '▷', running: '◐', done: '✓', unverified: '△', failed: '✕', skipped: '⊘', abandoned: '■',
}

function CallRow({ c, onDecide, onEndStep, othersRunning, now, runId }: {
  c: Call; onDecide?: (id: string, ok: boolean) => void; onEndStep?: (id: string) => void
  othersRunning?: boolean; now: number; runId?: string
}) {
  const [open, setOpen] = useState(false)
  const [full, setFull] = useState(false)
  const [confirmEnd, setConfirmEnd] = useState(false)
  const bad = c.state === 'failed'
  const warn = c.state === 'unverified'
  const arg = argLine(c.tool, c.args)
  const secs = c.t0 && c.t1 ? (c.t1 - c.t0) / 1000 : undefined
  const sub = c.tool === 'spawn_subagent'
  return (
    <div className={`border-l-2 pl-5 py-2.5 ${bad ? 'border-bad bg-bad/[0.05]' : warn || c.state === 'waiting' ? 'border-coral' : 'border-hairline'}`}>
      <div className="flex items-baseline gap-3 min-w-0">
        <span aria-hidden className={`num text-[14px] w-4 shrink-0 ${bad ? 'text-bad' : c.state === 'running' ? 'text-coral spin-slow inline-block' : warn || c.state === 'waiting' ? 'text-coral' : 'text-muted'}`}>
          {GLYPH[c.state]}
        </span>
        {c.agent !== 'main' && <span className="text-[13px] text-muted shrink-0">{c.agent} ›</span>}
        {!sub && DOES[c.tool] && <span className="text-[15px] text-ink shrink-0">{DOES[c.tool]}</span>}
        <span className={`num shrink-0 ${!sub && DOES[c.tool] ? 'text-[13px] text-muted' : 'text-[15px] text-ink'}`} translate="no">{c.tool}</span>
        <span className="sr-only">{c.state}</span>
        {c.detail && <span className={`text-[14px] truncate ${bad ? 'text-bad' : warn ? 'text-coral' : 'text-muted'}`} title={c.detail}>{c.detail}</span>}
        <span className="ml-auto num text-[13px] text-muted shrink-0">
          {c.state === 'running' ? `running ${c.t0 ? clock((now - c.t0) / 1000) : ''}` : secs != null ? clock(secs) : ''}
        </span>
      </div>
      {c.selfReview && (
        <p className="pl-7 mt-1.5 text-[14px] text-coral">
          {c.selfReview === 'no critic'
            ? 'The model wrote this review of its own work and filed it. No critic reached a verdict since the last review was filed, so nothing here is a second opinion.'
            : 'The critic’s last verdict was REJECTED. This review says APPROVED, and the model wrote it: it overrules the critic, it is not a second opinion.'}
        </p>
      )}
      {sub
        ? (
          <div className="pl-7 mt-1">
            <div className="text-[14px] text-body">Hand a task to a {String(c.args?.role || 'sub-agent')} (the same model, working on its own). {c.state === 'waiting' ? 'If you run this, its own steps will also ask you first.' : ''}</div>
            {c.state === 'waiting' && typeof c.args?.task === 'string' && (
              <Fold text={String(c.args.task)} lines={6} className="mt-1.5 text-[15px] leading-[1.55] text-ink2" />
            )}
          </div>
        )
        : arg && (
          <div className="pl-7 mt-1">
            <div className={`num text-[14px] text-body break-all whitespace-pre-wrap ${full ? '' : 'line-clamp-3'}`}>{arg}</div>
            {(arg.length > 300 || arg.split('\n').length > 3) && (
              <button onClick={() => setFull((f) => !f)} aria-expanded={full}
                      className="mt-1 text-[13px] text-muted hover:text-ink underline underline-offset-4">
                {full ? 'Show less' : c.tool === 'run_bash' || c.tool === 'Bash' ? 'Show the whole command' : 'Show all'}
              </button>
            )}
          </div>
        )}
      {c.state === 'running' && onEndStep && (
        <div className="pl-7 mt-2 flex items-center gap-2">
          {c.ending ? <span className="text-[13px] text-muted">Ending this step…</span>
            : confirmEnd ? (
              <>
                <span className="text-[14px] text-ink2">End this step and the processes it started? The run continues.{othersRunning ? ' Another step is running as well; if it started later, its processes end too.' : ''}</span>
                <button onClick={() => { onEndStep(c.id); setConfirmEnd(false) }}
                        className="h-8 px-3 rounded-[8px] bg-bad text-white text-[13px] font-semibold">End step</button>
                <button onClick={() => setConfirmEnd(false)} className="h-8 px-3 rounded-[8px] text-[13px] text-body hover:bg-card">Keep it running</button>
              </>
            ) : (
              <button onClick={() => setConfirmEnd(true)}
                      className="h-8 px-3 rounded-[8px] border line text-[13px] text-body hover:text-bad hover:border-bad/50">End this step</button>
            )}
        </div>
      )}
      {c.state === 'waiting' && onDecide && (
        <div className="pl-7 mt-2.5 flex items-center gap-2">
          <button onClick={() => onDecide(c.id, true)}
                  className="h-9 px-4 rounded-[8px] bg-coral text-on-coral text-[14px] font-semibold hover:bg-coral-h">Run this step</button>
          <button onClick={() => onDecide(c.id, false)}
                  className="h-9 px-4 rounded-[8px] border line text-[14px] text-body hover:text-ink hover:border-strong">Skip it</button>
          <span className="text-[13px] text-muted">openPASO is waiting for you.</span>
        </div>
      )}
      {c.result && !sub && (
        <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
                className="ml-7 mt-1.5 text-[13px] text-muted hover:text-ink underline underline-offset-4">
          {open ? 'Hide output' : 'Show output'}
        </button>
      )}
      {open && c.result && (
        <pre className="num text-[13px] leading-[1.6] text-body mt-2 ml-7 p-4 bg-soft border line rounded-[8px] max-h-[420px] overflow-auto scroll whitespace-pre-wrap break-words">
          {readable(c.result)}
        </pre>
      )}
      {runId && c.result && SOLVER_TOOLS.has(c.tool) && (c.state === 'done' || c.state === 'unverified' || c.state === 'failed') && (() => {
        const folder = stepFolder(readable(c.result), runId)
        return folder ? <StepOutputs runId={runId} sub={folder}
                                     verdict={c.state === 'done' ? 'verified' : c.state} /> : null
      })()}
    </div>
  )
}

function Fold({ text, lines, className }: { text: string; lines: number; className?: string }) {
  const [open, setOpen] = useState(false)
  const long = text.split('\n').length > lines + 1 || text.length > lines * 120
  return (
    <div className={className}>
      <div style={open || !long ? undefined
                  : { display: '-webkit-box', WebkitLineClamp: lines, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
        <Markdown text={text} />
      </div>
      {long && (
        <button onClick={() => setOpen((o) => !o)} aria-expanded={open}
                className="mt-1 text-[13px] text-muted hover:text-ink underline underline-offset-4">
          {open ? 'Show less' : 'Show all'}
        </button>
      )}
    </div>
  )
}

const ROLE: Record<string, string> = {
  critic: 'asked to find what is wrong with the work',
  verifier: 'asked to check a result another way',
  researcher: 'asked to look something up',
}
const roleName = (r: string) => r.charAt(0).toUpperCase() + r.slice(1)

const STEER: Record<string, string> = {
  queued: 'Waiting: openPASO reads it when the current step finishes. If the step hangs, End this step delivers it now.',
  queued_until_turn_ends: 'Waiting: Claude Code runs as one command and cannot be spoken to while it works, so this is sent the moment the turn ends.',
  delivered: 'Delivered to openPASO',
  sent_as_followup: 'The step had already finished, so it was sent as a follow-up',
  not_delivered: 'Not delivered: the run was stopped first',
}

/** What is happening between visible steps, so a clock never stands alone. */
function activity(entries: Entry[], events: Ev[], now: number): { text: string; warn: boolean } | null {
  const running = entries.find((e) => e.kind === 'call' && e.state === 'running') as Call | undefined
  if (running) {
    const s = running.t0 ? (now - running.t0) / 1000 : 0
    return s > 900
      ? { text: `${running.tool} has been running for ${clock(s)}. Long solves are normal; Stop ends it if it should not take this long.`, warn: false }
      : null
  }
  if (entries.some((e) => e.kind === 'call' && e.state === 'waiting')) return null
  const last = [...events].reverse().find((e) => e.type !== 'token_count' && e.type !== 'agent_chunk' && e.t)
  const any = [...events].reverse().find((e) => e.t)
  const since = any?.t ? (now - any.t) / 1000 : 0
  if (since > 300) {
    return { text: `No activity for ${clock(since)}. The model may be slow or stuck. You can send a correction or stop the run.`, warn: true }
  }
  const lastT = last?.t ? (now - last.t) / 1000 : 0
  if (!last || last.type === 'turn_start' || last.type === 'user_msg' || last.type === 'file_uploaded')
    return { text: `Starting openPASO and the model… ${clock(lastT)}`, warn: false }
  if (events[events.length - 1]?.type === 'agent_chunk') return { text: 'Writing…', warn: false }
  const inAside = entries.some((e) => e.kind === 'aside' && !e.returned)
  return { text: `${inAside ? 'The sub-agent is thinking…' : 'Thinking about the next step…'} ${clock(lastT)}`, warn: false }
}

export default function Transcript({ events, live, showReasoning, onDecide, onEndStep, modelKind, now = Date.now(), runId }: {
  events: Ev[]; live: boolean; showReasoning: boolean
  onDecide?: (id: string, ok: boolean) => void; onEndStep?: (id: string) => void; modelKind?: string; now?: number
  runId?: string
}) {
  const entries = useMemo(() => build(events, live), [events, live])
  const running = entries.filter((e) => e.kind === 'call' && e.state === 'running').length
  const [trace, setTrace] = useState<number | null>(null)
  const act = live ? activity(entries, events, now) : null

  if (!entries.length) {
    return <p className="text-[15px] text-muted py-6">{live ? 'Starting…' : 'Loading the run…'}</p>
  }

  return (
    <ol className="flex flex-col gap-1.5" aria-label="Run transcript" aria-live="polite">
      {modelKind === 'test' && (
        <li className="mb-2 rounded-[8px] px-5 py-3 border border-bad/40 bg-bad/[0.06] text-[15px] text-ink">
          This run used the fake test model. It runs no solver, and nothing in it is a result.
        </li>
      )}
      {entries.map((e, i) => {
        switch (e.kind) {
          case 'turn':
            return (
              <li key={i} className="flex items-center gap-3 pt-6 pb-2" aria-hidden>
                <span className="h-px flex-1 bg-hairline" />
                <span className="text-[13px] text-muted">Follow-up</span>
                <span className="h-px flex-1 bg-hairline" />
              </li>
            )
          case 'you':
            return (
              <li key={i} className="border-l-2 border-coral pl-5 py-3">
                <div className="text-[13px] font-semibold text-coral">You</div>
                <div className="mt-1 text-[17px] leading-[1.55] text-ink whitespace-pre-wrap">{e.text}</div>
                {e.files.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {e.files.map((f) => <span key={f} className="num h-7 px-2.5 rounded-[6px] bg-card text-[13px] text-body grid place-items-center">uploads/{f}</span>)}
                  </div>
                )}
              </li>
            )
          case 'upload':
            return (
              <li key={i} className="border-l-2 border-hairline pl-5 py-2 text-[14px] text-muted">
                You added <span className="num text-body">uploads/{e.name}</span> ({Math.max(1, Math.round(e.bytes / 1024))} KB)
              </li>
            )
          case 'thought':
            if (!e.final && !showReasoning) return null
            return (
              <li key={i} className={`border-l-2 pl-5 py-2.5 ${e.final ? 'border-ink2' : 'border-hairline'}`}>
                {e.final && (
                  <div className="mb-1 flex items-baseline gap-2 flex-wrap">
                    <span className="text-[13px] font-semibold text-ink2">Reply</span>
                    {e.outcome && e.outcome !== 'completed' && (
                      // a reply is prose whatever happened; where nothing was computed,
                      // say so where it is read, not only underneath it
                      <span className="text-[13px] text-coral">
                        {e.outcome === 'unverified'
                          ? 'a solver ran, but openPASO did not verify its result'
                          : "no solver result in this turn — any numbers below are the model's own"}
                      </span>
                    )}
                    {e.typed && e.typed.length > 0 && (
                      // a step typed into the reply looks like one that ran, result and all
                      <span className="text-[13px] text-coral basis-full">
                        {e.typed.length === 1
                          ? `This turn wrote a ${e.typed[0]} step out as text. It never ran; the result shown with it came from no tool.`
                          : `This turn wrote ${e.typed.length} steps out as text (${Array.from(new Set(e.typed)).join(', ')}). None of them ran; the results shown with them came from no tool.`}
                      </span>
                    )}
                  </div>
                )}
                <Fold text={e.text} lines={e.final ? 40 : 4}
                      className={e.final ? 'text-[16px] leading-[1.65] text-ink2' : 'text-[15px] leading-[1.6] text-body'} />
              </li>
            )
          case 'call':
            return (
              <li key={e.id || i}>
                <CallRow c={e} onDecide={onDecide} onEndStep={live ? onEndStep : undefined}
                         othersRunning={running > 1} now={now} runId={runId} />
              </li>
            )
          case 'aside':
            return (
              <li key={i} className="border-l-2 border-hairline pl-5 py-3 bg-soft/60 rounded-r-[8px]">
                <div className="text-[13px] font-semibold text-ink2">
                  {roleName(e.role)} <span className="font-normal text-muted">· the same model, {ROLE[e.role] || 'given a separate task'}; not an independent check</span>
                </div>
                {showReasoning && <Fold text={e.task} lines={3} className="mt-1 text-[15px] leading-[1.55] text-body" />}
              </li>
            )
          case 'verdict':
            return (
              <li key={i} className={`border-l-2 pl-5 py-3 rounded-r-[8px] bg-soft/60 ${e.reached ? 'border-ink2' : 'border-bad'}`}>
                <div className={`text-[13px] font-semibold ${e.reached ? 'text-ink2' : 'text-bad'}`}>
                  {e.reached ? `What the ${e.role} concluded` : `The ${e.role} stopped without a conclusion`}
                </div>
                {e.text && (
                  <Fold text={e.text} lines={showReasoning ? (e.reached ? 8 : 2) : 1}
                        className="mt-1 text-[15px] leading-[1.55] text-ink2" />
                )}
              </li>
            )
          case 'steer':
            return (
              <li key={i} className="border-l-2 border-coral pl-5 py-3 bg-coral/[0.04] rounded-r-[8px]">
                <div className="text-[13px] font-semibold text-coral">You · correction sent while it worked</div>
                <div className="mt-1 text-[16px] text-ink whitespace-pre-wrap">{e.text}</div>
                <div className={`mt-1.5 text-[13px] ${e.state === 'not_delivered' ? 'text-bad' : 'text-muted'}`}>{STEER[e.state] || e.state}</div>
              </li>
            )
          case 'mode':
            return (
              <li key={i} className="pl-5 py-1.5 text-[13px] text-muted">
                Steps changed to “{e.mode === 'plan' ? 'Ask before each step' : 'Run without asking'}”
              </li>
            )
          case 'end': {
            const bad = e.outcome === 'failed'
            const warn = e.outcome === 'unverified'
            const advice = bad && e.message ? ADVICE.find(([re]) => re.test(e.message!))?.[1] : undefined
            return (
              <li key={i} className={`mt-2 mb-2 rounded-[8px] px-5 py-3.5 border ${bad ? 'border-bad/40 bg-bad/[0.06]' : warn ? 'border-coral/40 bg-coral/[0.04]' : 'line bg-soft'}`}>
                <p className={`text-[15px] ${bad ? 'text-ink' : 'text-ink2'}`}>{endText(e)}</p>
                {e.tally.why && (
                  // the solver's own account, whole and folded rather than cut
                  // to a first sentence: the finding that explains a ladder
                  // ("your own logs imply a mesh ladder that was not halved")
                  // sat three thousand characters in, and an excerpt lost it
                  <Fold text={e.tally.why} lines={3}
                        className="mt-2 text-[15px] leading-[1.55] text-ink2" />
                )}
                {bad && e.message && <p className="mt-2 text-[15px] text-body break-words">{tidy(e.message)}</p>}
                {advice && <p className="mt-2 text-[15px] text-ink2">{advice}</p>}
                {e.left ? <p className="mt-2 text-[14px] text-bad">{e.left} process{e.left > 1 ? 'es' : ''} could not be ended; check the machine.</p> : null}
                {bad && e.traceback && (
                  <>
                    <button onClick={() => setTrace(trace === i ? null : i)} aria-expanded={trace === i}
                            className="mt-2 text-[13px] text-muted hover:text-ink underline underline-offset-4">
                      {trace === i ? 'Hide technical detail' : 'Show technical detail'}
                    </button>
                    {trace === i && <pre className="num mt-2 text-[13px] text-muted whitespace-pre-wrap max-h-[300px] overflow-auto scroll">{tidy(e.traceback)}</pre>}
                  </>
                )}
              </li>
            )
          }
        }
        return null
      })}
      {act && (
        <li className={`pl-5 py-3 flex items-center gap-3 text-[15px] ${act.warn ? 'text-bad' : 'text-body'}`} aria-live="polite">
          {!act.warn && <span aria-hidden className="text-coral spin-slow inline-block">◐</span>}
          <span>{act.text}</span>
        </li>
      )}
    </ol>
  )
}

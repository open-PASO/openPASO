export type Ev = {
  type: string
  cost_usd_total?: number
  /** the server's own verdict on a solver result, decided once */
  verdict?: string
  /** the solver's own account of its verdict */
  why?: string
  /** marks an ending the server decided, rather than one an old record claimed */
  by?: string
  text?: string
  tool?: string
  args?: Record<string, unknown>
  result?: string
  call_id?: string
  agent?: string
  role?: string
  task?: string
  context?: string
  message?: string
  input?: number
  output?: number
  reason?: string
  error?: string
  traceback?: string
  outcome?: string
  /** steps the turn's messages wrote out as text; none of them ran */
  typed_steps?: string[]
  sa_id?: string
  id?: string
  state?: string
  name?: string
  bytes?: number
  attachments?: string[]
  mode?: string
  model?: string
  usd?: number
  processes_ended?: number
  processes_left?: number
  t?: number
  seq?: number
}

export type Outcome =
  | 'running' | 'completed' | 'unverified' | 'no_result' | 'failed' | 'interrupted' | 'unfinished'

export type Session = {
  id: string
  created_at: number
  model: string
  mode: string
  mcp_servers: string[]
  tokens_in: number
  tokens_out: number
  cost_usd?: number
  model_detail?: string
  claude_model?: string
  running: boolean
  waiting?: boolean
  outcome: Outcome
  pending_steers: number
}

export type RunRow = {
  id: string
  created_at: number
  updated_at: number
  model: string
  model_label: string
  model_kind: string
  model_detail?: string | null
  mode: string
  prompt: string | null
  outcome: Outcome
  running: boolean
  waiting?: boolean
  steps: number
  cost_usd?: number | null
}

export type ModelInfo = {
  id: string
  label: string
  kind: 'openrouter' | 'claude-code' | 'local'
  available: boolean
  status: string
  price_in?: number | null
  price_out?: number | null
  plan_mode?: boolean
  past_runs?: number
  past_cost_low?: number | null
  past_cost_high?: number | null
}

export type ModelGroup = {
  kind: string
  title: string
  note: string
  key_source?: string | null
  models: ModelInfo[]
}

export type ModeInfo = { id: 'plan' | 'accept'; label: string; detail: string }

export type AppConfig = {
  /** the bundle this server serves; a page running another one is out of date */
  build?: string | null
  modes: ModeInfo[]
  default_mode: string
  docs_url: string
  max_running: number
}

export type SolverCheck = {
  ok: boolean
  error?: string
  checked_at: number
  mesher?: boolean | null
  python?: string
  solvers: Solver[]
}

export type Solver = {
  name: string
  status: string
  version: string | null
  physics: number | null
}

export type FileRow = {
  name: string
  rel_path: string
  sub: string
  is_dir: boolean
  size: number | null
  mtime: number
  kind: string
}

export type FieldSeries = {
  kind: 'field_series'
  url: string
  field: string
  unit: string
  nx: number
  ny: number
  vmin: number
  vmax: number
  n_frames: number
  x0?: number
  y0?: number
  dx?: number
  dy?: number
  provenance?: {
    true_min?: number
    true_max?: number
    clip_percentile?: number | null
    saturated_fraction?: number
    quantisation_step?: number
    interpolation?: string
    solver?: string | null
    source?: string | null
    sha256?: string
  }
}

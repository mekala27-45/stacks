/** Portable Fetch + D1 kernel. No provider SDK or filesystem dependency. */
export interface SqlResult<T = Record<string, unknown>> { results: T[]; meta: { changes?: number } }
export interface SqlStatement {
  bind(...values: unknown[]): SqlStatement;
  first<T = Record<string, unknown>>(column?: string): Promise<T | null>;
  all<T = Record<string, unknown>>(): Promise<SqlResult<T>>;
  run(): Promise<SqlResult>;
}
export interface SqlDatabase { prepare(query: string): SqlStatement; batch(statements: SqlStatement[]): Promise<SqlResult[]> }
export interface Env {
  DB: SqlDatabase;
  ASSETS: { fetch(request: Request): Promise<Response> };
  SESSION_HASH_KEY: string;
  CORS_ORIGINS?: string;
  ADMIN_TOKEN?: string;
  DATA_PREFIX?: string;
}
type Model = 'als' | 'blend' | 'popularity' | 'item_cosine';
type Book = { id: string | number; title: string; author: string; genre: string; tags: string[]; popularity: number; year?: number | null; available?: boolean };
type Profile = { id: string | number; positive_indices: number[]; seen_indices: number[]; user_factor: number[] };
type Manifest = { schema_version: string; artifact_version: string; catalog_size: number; factors: number; alpha: number; regularization: number; files?: Record<string, string>; sha256?: Record<string, string> };
type Data = { manifest: Manifest; books: Book[]; profiles: Map<string, Profile>; popularity: Float32Array; factors: Float32Array; cosine: Float32Array; indices: Uint32Array; indptr: Uint32Array; gram?: Float64Array };
type SessionRow = { session_hash: string; created_at: string; updated_at: string; arm: string; reader_id: string | null; history: string; seen: string; genre: string; revision: number; operation_id: string; latest_shelf: string };
type ImpressionRow = { id: string; session_hash: string; item_id: string; position: number; propensity: number; arm: string; model_version: string; impression_at: string; policy: string; candidate_pool: string; trace: string };
type FeedbackRow = { id: string; impression_id: string; event: string; rating: number | null; feedback_at: string };
type Scored = Book & { index: number; score: number; scores: Record<string, number> };
type Exploration = { selected: boolean; policy: string; candidate_pool: string[]; propensity: number; target_policy: string; target_probabilities: number[]; reward_model: number[]; reward_horizon_seconds: number; support: string };
type Trace = { traffic_kind?: string; retrieval: Record<string, unknown>; ranking: Record<string, unknown>; reranking: Record<string, unknown>; exploration: Exploration; shadow: Record<string, unknown> };
type Item = Omit<Scored, 'index' | 'scores'> & { id: string; impression_id: string; position: number; propensity: number; explanation: string; trace: Trace };
export type Shelf = { schema_version: string; revision: number; model_version: string; artifact_version: string; scoring_mode: string; arm: string; experiment_id: string; backend: string; items: Item[]; history: string[]; genre: string; traffic_kind: string; statement: string; latency_ms: number };
type Estimate = { mean: number; low: number; high: number };

const STATEMENT = "These are demonstration recommendations on a public dataset; no real reader's identity is present and no recommendation is personalized to a real person.";
const EXPERIMENT = 'evaluated-als-blend-v2';
const TARGET = 'final-slot-80-best-20-uniform-v1';
const HORIZON = 60;
const encoder = new TextEncoder();
let cached: Promise<Data> | undefined;

class HttpError extends Error { constructor(public status: number, message: string) { super(message); } }
function fail(status: number, message: string): never { throw new HttpError(status, message); }
function json(value: unknown, status = 200): Response { return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' } }); }
function array<T>(text: string): T[] { return JSON.parse(text) as T[]; }
function digestHex(bytes: ArrayBuffer): string { return [...new Uint8Array(bytes)].map(v => v.toString(16).padStart(2, '0')).join(''); }
async function hash(value: string, key: string): Promise<string> {
  if (!key || key.length < 16) fail(503, 'Configure a persistent SESSION_HASH_KEY before serving sessions');
  const imported = await crypto.subtle.importKey('raw', encoder.encode(key), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return digestHex(await crypto.subtle.sign('HMAC', imported, encoder.encode(value)));
}
async function armFor(hashed: string): Promise<string> {
  const bytes = new Uint8Array(await crypto.subtle.digest('SHA-256', encoder.encode(`${EXPERIMENT}:${hashed}`)));
  let value = 0n; for (const byte of bytes.slice(0, 8)) value = (value << 8n) + BigInt(byte);
  return value % 100n < 50n ? 'als' : 'blend';
}
function token(): string { return digestHex(crypto.getRandomValues(new Uint8Array(32)).buffer); }
function uniformIndex(size: number): number {
  const limit = Math.floor(0x100000000 / size) * size;
  let value: number; do { value = crypto.getRandomValues(new Uint32Array(1))[0]; } while (value >= limit);
  return value % size;
}
async function loadData(env: Env, origin: string): Promise<Data> {
  if (!cached) cached = (async () => {
    const prefix = env.DATA_PREFIX || '/data/';
    let checksums: Record<string, string> = {};
    const asset = async (name: string): Promise<Response> => {
      if (name.includes('..') || name.startsWith('/') || name.includes('://')) throw new Error('Invalid artifact asset path');
      const result = await env.ASSETS.fetch(new Request(new URL(prefix + name, origin)));
      if (!result.ok) throw new Error(`Model asset unavailable: ${name}`);
      if (checksums[name]) {
        const bytes = await result.arrayBuffer();
        if (digestHex(await crypto.subtle.digest('SHA-256', bytes)) !== checksums[name]) throw new Error(`Artifact checksum mismatch: ${name}`);
        return new Response(bytes, { headers: result.headers });
      }
      return result;
    };
    const manifest = await (await asset('edge-manifest.json')).json() as Manifest;
    if (!Number.isInteger(manifest.catalog_size) || manifest.catalog_size < 1 || manifest.catalog_size > 100000 || !Number.isInteger(manifest.factors) || manifest.factors < 1 || manifest.factors > 256 || !Number.isFinite(manifest.alpha) || manifest.alpha <= 0 || !Number.isFinite(manifest.regularization) || manifest.regularization <= 0 || typeof manifest.artifact_version !== 'string' || !manifest.artifact_version) throw new Error('Invalid artifact metadata');
    checksums = manifest.sha256 || {};
    const files = manifest.files || {};
    const [books, profiles, popularity, factors, cosine, indices, indptr] = await Promise.all([
      asset(files.catalog || 'catalog.json').then(r => r.json() as Promise<Book[]>),
      asset(files.profiles || 'edge-profiles.json').then(r => r.json() as Promise<Profile[]>),
      asset(files.popularity || 'popularity.f32').then(r => r.arrayBuffer()),
      asset(files.item_factors || 'item-factors.f32').then(r => r.arrayBuffer()),
      asset(files.cosine_data || 'cosine-data.f32').then(r => r.arrayBuffer()),
      asset(files.cosine_indices || 'cosine-indices.u32').then(r => r.arrayBuffer()),
      asset(files.cosine_indptr || 'cosine-indptr.u32').then(r => r.arrayBuffer()),
    ]);
    if (books.length !== manifest.catalog_size || factors.byteLength !== books.length * manifest.factors * 4 || popularity.byteLength !== books.length * 4 || indptr.byteLength !== (books.length + 1) * 4) throw new Error('Artifact shape mismatch');
    const loaded: Data = { manifest, books, profiles: new Map(profiles.map(p => [String(p.id), p])), popularity: new Float32Array(popularity), factors: new Float32Array(factors), cosine: new Float32Array(cosine), indices: new Uint32Array(indices), indptr: new Uint32Array(indptr) };
    if (books.some((book, index) => Number(book.id) !== index + 1) || loaded.factors.some(value => !Number.isFinite(value)) || loaded.popularity.some(value => !Number.isFinite(value) || value < 0) || loaded.cosine.some(value => !Number.isFinite(value) || value < 0 || value > 1.000001) || loaded.cosine.length !== loaded.indices.length || loaded.indices.some(value => value >= books.length) || loaded.indptr[0] !== 0 || loaded.indptr.at(-1) !== loaded.cosine.length || loaded.indptr.some((value, index) => index > 0 && value < loaded.indptr[index - 1])) throw new Error('Invalid binary model values');
    if (profiles.some(profile => profile.user_factor.length !== manifest.factors || profile.user_factor.some(value => !Number.isFinite(value)) || [...profile.positive_indices, ...profile.seen_indices].some(value => !Number.isInteger(value) || value < 0 || value >= books.length))) throw new Error('Invalid reader artifact profile');
    return loaded;
  })().catch(error => { cached = undefined; throw error; });
  return cached;
}
function normalize(values: ArrayLike<number>): Float64Array {
  let low = Infinity, high = -Infinity;
  for (let i = 0; i < values.length; i++) { low = Math.min(low, values[i]); high = Math.max(high, values[i]); }
  return Float64Array.from(values, value => high > low ? (value - low) / (high - low) : 0);
}
function solve(matrix: Float64Array, rhs: Float64Array, n: number): Float64Array {
  // Partial-pivot Gaussian elimination of the positive-definite ALS normal equations.
  const a = matrix.slice(), b = rhs.slice();
  for (let column = 0; column < n; column++) {
    let pivot = column;
    for (let row = column + 1; row < n; row++) if (Math.abs(a[row * n + column]) > Math.abs(a[pivot * n + column])) pivot = row;
    if (Math.abs(a[pivot * n + column]) < 1e-12) throw new Error('Singular ALS fold-in system');
    if (pivot !== column) { for (let j = column; j < n; j++) [a[pivot * n + j], a[column * n + j]] = [a[column * n + j], a[pivot * n + j]]; [b[pivot], b[column]] = [b[column], b[pivot]]; }
    for (let row = column + 1; row < n; row++) {
      const factor = a[row * n + column] / a[column * n + column];
      for (let j = column; j < n; j++) a[row * n + j] -= factor * a[column * n + j];
      b[row] -= factor * b[column];
    }
  }
  const result = new Float64Array(n);
  for (let row = n - 1; row >= 0; row--) { let value = b[row]; for (let j = row + 1; j < n; j++) value -= a[row * n + j] * result[j]; result[row] = value / a[row * n + row]; }
  return result;
}
function foldIn(data: Data, history: number[]): Float64Array {
  const f = data.manifest.factors, alpha = data.manifest.alpha ?? 20, regularization = data.manifest.regularization ?? 0.1;
  if (!data.gram) {
    data.gram = new Float64Array(f * f);
    for (let k = 0; k < f; k++) data.gram[k * f + k] = regularization;
    for (let item = 0; item < data.books.length; item++) for (let a = 0; a < f; a++) for (let b = 0; b < f; b++) data.gram[a * f + b] += data.factors[item * f + a] * data.factors[item * f + b];
  }
  const normal = data.gram.slice(), rhs = new Float64Array(f);
  for (const item of history) for (let a = 0; a < f; a++) {
    rhs[a] += (1 + alpha) * data.factors[item * f + a];
    for (let b = 0; b < f; b++) normal[a * f + b] += alpha * data.factors[item * f + a] * data.factors[item * f + b];
  }
  return solve(normal, rhs, f);
}
export function scoreArtifacts(data: Data, historyIds: string[], readerId: string | null): { rows: Scored[]; mode: string } {
  const history = [...new Set(historyIds.map(id => Number(id) - 1))].filter(i => i >= 0 && i < data.books.length);
  const profile = readerId ? data.profiles.get(readerId) : undefined;
  const exact = profile && history.length === profile.positive_indices.length && profile.positive_indices.every(i => history.includes(i));
  const factors = exact ? Float64Array.from(profile.user_factor) : history.length ? foldIn(data, history) : new Float64Array(data.manifest.factors);
  const cosine = new Float64Array(data.books.length), als = new Float64Array(data.books.length);
  if (!history.length) { cosine.set(data.popularity); als.set(data.popularity); }
  else {
    for (const item of history) for (let offset = data.indptr[item]; offset < data.indptr[item + 1]; offset++) cosine[data.indices[offset]] += data.cosine[offset];
    for (let item = 0; item < data.books.length; item++) for (let f = 0; f < factors.length; f++) als[item] += data.factors[item * factors.length + f] * factors[f];
  }
  const nc = normalize(cosine), na = normalize(als), np = normalize(data.popularity);
  const rows = data.books.map((book, index) => ({ ...book, index, score: 0, scores: { popularity: data.popularity[index], item_cosine: cosine[index], als: als[index], blend: .55 * nc[index] + .35 * na[index] + .10 * np[index] } }));
  return { rows, mode: exact ? 'evaluated-reader-exact' : history.length ? 'frozen-item-factor-session-fold-in' : 'cold-popularity' };
}
function overlap(a: Book, b: Book): number {
  const x = new Set(a.tags), y = new Set(b.tags); if (!x.size || !y.size) return Number(a.genre === b.genre);
  return [...x].filter(tag => y.has(tag)).length / new Set([...x, ...y]).size;
}
function makeShelf(data: Data, session: SessionRow, limit = 10): { shelf: Shelf; impressions: ImpressionRow[] } {
  const start = performance.now(), history = array<string>(session.history), seen = new Set(array<string>(session.seen));
  const trafficKind = (JSON.parse(session.latest_shelf) as Partial<Shelf>).traffic_kind || 'interactive';
  const scored = scoreArtifacts(data, history, session.reader_id), model: Model = session.arm === 'als' ? 'als' : 'blend';
  const sorted = scored.rows.map(row => ({ ...row, score: row.scores[model] })).filter(row => !seen.has(String(row.id)) && row.available !== false && (session.genre === 'All books' || row.genre === session.genre)).sort((a, b) => b.score - a.score || a.index - b.index);
  const candidates: Scored[] = [], authors = new Map<string, number>();
  for (const row of sorted.slice(0, 200)) { const author = row.author.split(',')[0].trim().toLowerCase(); if ((authors.get(author) || 0) < 2) { candidates.push(row); authors.set(author, (authors.get(author) || 0) + 1); } }
  const selected = candidates.slice(0, Math.max(0, limit - 1));
  const pool = candidates.filter(row => !selected.includes(row)).slice(0, 20);
  const exploration = pool.length ? pool[uniformIndex(pool.length)] : undefined;
  if (exploration) selected.push(exploration);
  const shadowModel = model === 'als' ? 'blend' : 'als';
  const shadow = scored.rows.filter(row => !seen.has(String(row.id)) && (session.genre === 'All books' || row.genre === session.genre)).sort((a, b) => b.scores[shadowModel] - a.scores[shadowModel] || a.index - b.index).slice(0, limit);
  const disagreement = 1 - selected.filter(row => shadow.some(other => row.id === other.id)).length / Math.max(selected.length, 1);
  const now = new Date().toISOString(), impressions: ImpressionRow[] = [];
  const items: Item[] = selected.map((row, offset) => {
    const exploring = row === exploration, support = exploring ? pool : [row], propensity = exploring ? 1 / pool.length : 1;
    const trace: Trace = {
      traffic_kind: trafficKind,
      retrieval: { backend: 'evaluated-artifacts', candidate_count: Math.min(sorted.length, 200), top_candidates: sorted.slice(0, 12).map(value => ({ id: String(value.id), score: value.score })) },
      ranking: { model_version: `${model}:${data.manifest.artifact_version}`, score: row.score, features: row.scores, weights: model === 'blend' ? { item_cosine: .55, als: .35, popularity: .10 } : { als: 1 }, scoring_mode: scored.mode },
      reranking: { rules: { author_cap: 2, genre: session.genre }, diversity: { weight: 0, method: 'disabled to preserve evaluated score ordering' }, calibration: { weight: 0 }, before_ids: sorted.slice(0, limit).map(value => String(value.id)) },
      exploration: { selected: exploring, policy: exploring ? 'uniform' : 'deterministic', candidate_pool: support.map(value => String(value.id)), propensity, target_policy: TARGET, target_probabilities: support.map((_, i) => .2 / support.length + (i === 0 ? .8 : 0)), reward_model: support.map(() => .05), reward_horizon_seconds: HORIZON, support: 'Conditional final-position action distribution on this fixed pool; no full-slate OPE' },
      shadow: { model_version: `${shadowModel}:${data.manifest.artifact_version}`, top_ids: shadow.map(value => String(value.id)), top_scores: shadow.map(value => ({ id: String(value.id), score: value.scores[shadowModel] })), set_disagreement: disagreement, served: false },
    };
    const id = crypto.randomUUID();
    impressions.push({ id, session_hash: session.session_hash, item_id: String(row.id), position: offset + 1, propensity, arm: session.arm, model_version: `${model}:${data.manifest.artifact_version}`, impression_at: now, policy: exploring ? 'uniform' : 'deterministic', candidate_pool: JSON.stringify(trace.exploration.candidate_pool), trace: JSON.stringify(trace) });
    const { index: _index, scores: _scores, ...book } = row;
    return { ...book, id: String(row.id), impression_id: id, position: offset + 1, propensity, explanation: exploring ? 'Uniform exploration within a recorded candidate pool. Its probability and fixed click horizon are stored.' : scored.mode === 'cold-popularity' ? 'Training-reader favorites while this demonstration has no reading history.' : `Ranked by the evaluated ${model === 'als' ? 'implicit ALS' : 'fixed cosine, ALS and popularity blend'} artifacts.`, trace };
  });
  return { shelf: { schema_version: '1.1', revision: session.revision + 1, model_version: `${model}:${data.manifest.artifact_version}`, artifact_version: data.manifest.artifact_version, scoring_mode: scored.mode, arm: session.arm, experiment_id: EXPERIMENT, backend: 'live-edge-d1', items, history, genre: session.genre, traffic_kind: trafficKind, statement: STATEMENT, latency_ms: performance.now() - start }, impressions };
}

async function getSession(env: Env, raw: string): Promise<SessionRow> {
  if (!/^[a-f0-9]{64}$/.test(raw)) fail(404, 'Session not found');
  const result = await env.DB.prepare('SELECT * FROM stacks_sessions WHERE session_hash=?').bind(await hash(raw, env.SESSION_HASH_KEY)).first<SessionRow>();
  return result || fail(404, 'Session not found');
}
async function writeShelf(env: Env, session: SessionRow, generated: { shelf: Shelf; impressions: ImpressionRow[] }, isNew: boolean, feedback?: FeedbackRow): Promise<void> {
  const operation = crypto.randomUUID(), now = new Date().toISOString(), statements: SqlStatement[] = [];
  if (isNew) statements.push(env.DB.prepare('INSERT INTO stacks_sessions(session_hash,created_at,updated_at,arm,reader_id,history,seen,genre,revision,operation_id,latest_shelf) VALUES(?,?,?,?,?,?,?,?,?,?,?)').bind(session.session_hash, now, now, session.arm, session.reader_id, session.history, session.seen, session.genre, generated.shelf.revision, operation, JSON.stringify(generated.shelf)));
  else statements.push(env.DB.prepare('UPDATE stacks_sessions SET updated_at=?,history=?,seen=?,genre=?,revision=?,operation_id=?,latest_shelf=? WHERE session_hash=? AND revision=?').bind(now, session.history, session.seen, session.genre, generated.shelf.revision, operation, JSON.stringify(generated.shelf), session.session_hash, session.revision));
  for (const row of generated.impressions) statements.push(env.DB.prepare('INSERT INTO stacks_impressions(id,session_hash,item_id,position,propensity,arm,model_version,impression_at,policy,candidate_pool,trace) SELECT ?,?,?,?,?,?,?,?,?,?,? WHERE EXISTS(SELECT 1 FROM stacks_sessions WHERE session_hash=? AND operation_id=?)').bind(row.id, row.session_hash, row.item_id, row.position, row.propensity, row.arm, row.model_version, row.impression_at, row.policy, row.candidate_pool, row.trace, session.session_hash, operation));
  if (feedback) statements.push(env.DB.prepare('INSERT INTO stacks_feedback(id,impression_id,event,rating,feedback_at) SELECT ?,?,?,?,? WHERE EXISTS(SELECT 1 FROM stacks_sessions WHERE session_hash=? AND operation_id=?)').bind(feedback.id, feedback.impression_id, feedback.event, feedback.rating, feedback.feedback_at, session.session_hash, operation));
  const results = await env.DB.batch(statements);
  if (!results[0].meta.changes) fail(409, 'Session changed concurrently; retry the request');
}
async function bounds(env: Env, session: SessionRow): Promise<void> {
  const count = await env.DB.prepare('SELECT COUNT(*) AS n FROM stacks_impressions WHERE session_hash=?').bind(session.session_hash).first<{ n: number }>();
  if ((count?.n || 0) >= 990) fail(429, 'This demonstration session reached its storage limit');
}
async function body(request: Request): Promise<Record<string, unknown>> {
  const reader = request.body?.getReader(), chunks: Uint8Array[] = []; let size = 0;
  if (reader) { try { while (true) { const { done, value } = await reader.read(); if (done) break; size += value.byteLength; if (size > 32768) { await reader.cancel(); fail(413, 'Request body too large'); } chunks.push(value); } } finally { reader.releaseLock(); } }
  const bytes = new Uint8Array(size); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  const content = new TextDecoder().decode(bytes);
  try { const parsed = JSON.parse(content) as unknown; if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) fail(422, 'Expected JSON object'); return parsed as Record<string, unknown>; } catch { return fail(422, 'Invalid JSON object'); }
}
function requireAdmin(request: Request, env: Env): void { if (!env.ADMIN_TOKEN) fail(503, 'Administrative access is not configured'); if (request.headers.get('X-Admin-Token') !== env.ADMIN_TOKEN) fail(401, 'Administrative token required'); }
function validateGenre(value: unknown, data: Data): string { if (value === undefined || value === 'All books') return 'All books'; if (typeof value !== 'string' || !data.books.some(book => book.genre === value)) fail(422, 'Unknown genre'); return value; }

export function evaluateLoggedRows(rows: ImpressionRow[], feedback: FeedbackRow[], now = Date.now()): Record<string, unknown> {
  const excludedTestRows = rows.filter(row => row.policy === 'uniform' && (JSON.parse(row.trace) as Trace).traffic_kind === 'load_test').length;
  const eligible = rows.filter(row => row.policy === 'uniform' && (JSON.parse(row.trace) as Trace).traffic_kind !== 'load_test');
  const observations: { w: number; reward: number; bound: number; id: string }[] = [];
  let pending = 0, invalid = 0, late = 0;
  for (const row of eligible) {
    const exploration = (JSON.parse(row.trace) as Trace).exploration;
    if (exploration.target_policy !== TARGET || exploration.candidate_pool.length !== exploration.target_probabilities.length || exploration.reward_horizon_seconds !== HORIZON) { invalid++; continue; }
    const end = Date.parse(row.impression_at) + HORIZON * 1000;
    if (now < end) { pending++; continue; }
    const index = exploration.candidate_pool.indexOf(row.item_id), n = exploration.candidate_pool.length;
    if (index < 0 || n < 1 || n > 20 || Math.abs(row.propensity - 1 / n) > 1e-9 || exploration.target_probabilities.some((p, i) => !Number.isFinite(p) || Math.abs(p - (.2 / n + (i === 0 ? .8 : 0))) > 1e-9)) { invalid++; continue; }
    const clicks = feedback.filter(event => event.impression_id === row.id && event.event === 'click');
    late += clicks.filter(event => Date.parse(event.feedback_at) > end).length;
    const reward = Number(clicks.some(event => Date.parse(event.feedback_at) >= Date.parse(row.impression_at) && Date.parse(event.feedback_at) <= end));
    observations.push({ w: exploration.target_probabilities[index] / row.propensity, reward, bound: Math.max(...exploration.target_probabilities) / row.propensity, id: row.session_hash });
  }
  const common = { schema_version: '1.1', target_policy: TARGET, scope: 'Conditional final-position action value within each recorded candidate pool', reward: 'Click received within the fixed exposure horizon', horizon_seconds: HORIZON, matured: observations.length, pending, invalid_excluded: invalid, load_test_excluded: excludedTestRows, late_feedback_ignored: late, session_count: new Set(observations.map(row => row.id)).size, full_slate_identified: false, statement: STATEMENT };
  if (!observations.length) return { ...common, status: 'awaiting_mature_feedback', estimates: null, detail: 'No randomized impression has completed its reward horizon.' };
  const n = observations.length, mean = (values: number[]): number => values.reduce((a, b) => a + b, 0) / n;
  // A predetermined maximum remains valid when prior feedback changes future pools.
  const bound = 16.2, weighted = mean(observations.map(row => row.w * row.reward)), denominator = mean(observations.map(row => row.w));
  const radius = bound * Math.sqrt(Math.log(40) / (2 * n)), ratioRadius = bound * Math.sqrt(Math.log(80) / (2 * n));
  const dr = .05 + mean(observations.map(row => row.w * (row.reward - .05)));
  const estimates: Record<string, Estimate> = {
    IPS: { mean: weighted, low: weighted - radius, high: weighted + radius },
    SNIPS: { mean: weighted / denominator, low: Math.max(0, weighted - ratioRadius) / Math.max(.2, denominator + ratioRadius), high: Math.min(1, (weighted + ratioRadius) / Math.max(.2, denominator - ratioRadius)) },
    DM: { mean: .05, low: .05, high: .05 },
    DR: { mean: dr, low: dr - radius, high: dr + radius },
  };
  return { ...common, status: 'measured', estimates, logging_click_rate: mean(observations.map(row => row.reward)), effective_sample_size: (n * denominator) ** 2 / observations.reduce((sum, row) => sum + row.w ** 2, 0), maximum_importance_weight: Math.max(...observations.map(row => row.w)), interval_method: 'Fixed-snapshot 95% bounded concentration intervals; conservative for adaptive session contexts. SNIPS uses simultaneous numerator/denominator bounds. DM is conditional on a fixed 0.05 prediction, not a learned causal interval.', detail: 'The target is 80% best scored remaining candidate plus 20% uniform. All target actions have positive logged support. These estimates describe this randomized position under the logged prefixes, not changing the deterministic shelf or its induced future traffic. Intervals are not a sequential stopping rule.' };
}
async function persisted(env: Env, sessionHash?: string): Promise<{ impressions: ImpressionRow[]; feedback: FeedbackRow[] }> {
  const query = sessionHash ? env.DB.prepare('SELECT * FROM stacks_impressions WHERE session_hash=? ORDER BY impression_at LIMIT 1000').bind(sessionHash) : env.DB.prepare('SELECT * FROM stacks_impressions ORDER BY impression_at DESC LIMIT 10000');
  const impressions = (await query.all<ImpressionRow>()).results;
  const events = sessionHash ? env.DB.prepare('SELECT f.* FROM stacks_feedback f JOIN stacks_impressions i ON i.id=f.impression_id WHERE i.session_hash=?').bind(sessionHash) : env.DB.prepare('SELECT * FROM stacks_feedback ORDER BY feedback_at DESC LIMIT 30000');
  return { impressions, feedback: (await events.all<FeedbackRow>()).results };
}
async function route(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url), path = url.pathname.replace(/\/$/, ''), parts = path.split('/').filter(Boolean);
  if (path === '/health') { const data = await loadData(env, url.origin); await env.DB.prepare('SELECT 1').first(); return json({ status: 'ok', backend: 'live-edge-d1', schema_version: '1.1', artifact_version: data.manifest.artifact_version, catalog_size: data.books.length, statement: STATEMENT }); }
  if (request.method === 'POST') {
    const length = Number(request.headers.get('Content-Length') || 0); if (length > 32768) fail(413, 'Request body too large');
    const identity = await hash(request.headers.get('CF-Connecting-IP') || 'unknown', env.SESSION_HASH_KEY), minute = Math.floor(Date.now() / 60000);
    const rate = await env.DB.prepare('INSERT INTO stacks_rate_limits(bucket,count,expires_at) VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET count=count+1 RETURNING count').bind(`${identity}:${minute}`, (minute + 2) * 60000).first<{ count: number }>();
    if ((rate?.count || 0) > 60) fail(429, 'Demo write limit reached; retry after one minute');
    await env.DB.prepare('DELETE FROM stacks_rate_limits WHERE expires_at<?').bind(Date.now()).run();
  }
  if (path === '/v1/catalog') return json({ items: (await loadData(env, url.origin)).books, statement: STATEMENT });
  if (path === '/v1/ope') { const logs = await persisted(env); return json(evaluateLoggedRows(logs.impressions, logs.feedback)); }
  if (path === '/v1/monitoring') {
    const raw = await persisted(env), mature = evaluateLoggedRows(raw.impressions, raw.feedback);
    const impressions = raw.impressions.filter(row => (JSON.parse(row.trace) as Trace).traffic_kind !== 'load_test');
    const ids = new Set(impressions.map(row => row.id));
    const logs = { impressions, feedback: raw.feedback.filter(row => ids.has(row.impression_id)) };
    const positions = [...new Set(logs.impressions.map(row => row.position))].sort((a, b) => a - b).map(position => { const exposures = logs.impressions.filter(row => row.position === position && Date.parse(row.impression_at) + HORIZON * 1000 <= Date.now()); const clicks = exposures.filter(row => logs.feedback.some(event => event.impression_id === row.id && event.event === 'click' && Date.parse(event.feedback_at) <= Date.parse(row.impression_at) + HORIZON * 1000)); return { position, matured_impressions: exposures.length, clicks: clicks.length, click_rate: exposures.length ? clicks.length / exposures.length : null }; });
    const firstPerShelf = logs.impressions.filter(row => row.position === 1), disagreements = firstPerShelf.map(row => Number((JSON.parse(row.trace) as Trace).shadow.set_disagreement));
    return json({ impressions: logs.impressions.length, feedback_events: logs.feedback.length, load_test_impressions_excluded: raw.impressions.length - logs.impressions.length, window: 'Interactive traffic within the latest 10000 persisted impressions', coverage_items: new Set(logs.impressions.map(row => row.item_id)).size, by_position: positions, shadow: { measured_shelves: disagreements.length, mean_set_disagreement: disagreements.length ? disagreements.reduce((a, b) => a + b, 0) / disagreements.length : null }, ope: mature, online_lift: null, note: 'Position CTR is observational and reflects position bias. Load-test traffic is excluded from these statistics. No online lift is claimed.', statement: STATEMENT });
  }
  if (path === '/v1/registry') return json({ status: 'reference_registry', activation_available: false, detail: 'Cloudflare serves versioned evaluated ALS/blend artifacts; evidence promotion gates are executed by the Python reference API and CI. This edge endpoint cannot activate unverified artifacts.', statement: STATEMENT });
  if (path === '/v1/session' && request.method === 'POST') {
    const data = await loadData(env, url.origin), input = await body(request), reader = input.reader_id === undefined || input.reader_id === null ? null : String(input.reader_id), profile = reader ? data.profiles.get(reader) : undefined;
    if (reader && !profile) fail(404, 'Public sample reader not found');
    const total = await env.DB.prepare('SELECT COUNT(*) AS n FROM stacks_sessions').first<{ n: number }>(); if ((total?.n || 0) >= 500) fail(429, 'Demonstration session cap reached');
    const raw = token(), hashed = await hash(raw, env.SESSION_HASH_KEY), now = new Date().toISOString();
    if (input.traffic_kind !== undefined && input.traffic_kind !== 'interactive' && input.traffic_kind !== 'load_test') fail(422, 'Unknown traffic kind');
    const session: SessionRow = { session_hash: hashed, created_at: now, updated_at: now, arm: await armFor(hashed), reader_id: reader, history: JSON.stringify(profile?.positive_indices.map(i => String(i + 1)) || []), seen: JSON.stringify(profile?.seen_indices.map(i => String(i + 1)) || []), genre: validateGenre(input.genre, data), revision: 0, operation_id: '', latest_shelf: JSON.stringify({ traffic_kind: input.traffic_kind || 'interactive' }) };
    const generated = makeShelf(data, session); await writeShelf(env, session, generated, true); return json({ session_id: raw, ...generated.shelf }, 201);
  }
  if (parts[0] === 'v1' && parts[1] === 'session' && parts[2]) {
    const raw = parts[2], session = await getSession(env, raw), action = parts[3];
    if (!action && request.method === 'GET') return json({ session_id: raw, ...JSON.parse(session.latest_shelf) as Shelf });
    if (action === 'logs' && request.method === 'GET') { const logs = await persisted(env, session.session_hash); return json({ schema_version: '1.1', impressions: logs.impressions.map(({ session_hash: _hash, ...row }) => ({ ...row, candidate_pool: array<string>(row.candidate_pool), trace: JSON.parse(row.trace) as unknown })), feedback: logs.feedback, statement: STATEMENT }); }
    if (action === 'ope' && request.method === 'GET') { const logs = await persisted(env, session.session_hash); return json(evaluateLoggedRows(logs.impressions, logs.feedback)); }
    if (action === 'stream' && request.method === 'GET') {
      const initial = Number(request.headers.get('Last-Event-ID') ?? -1); if (!Number.isInteger(initial)) fail(400, 'Last-Event-ID must be a revision');
      const once = url.searchParams.get('once') === 'true'; let revision = initial, cancelled = false;
      const stream = new ReadableStream<Uint8Array>({ async start(controller) { try { for (let turn = 0; !cancelled && !request.signal.aborted && turn < 110; turn++) { const latest = await getSession(env, raw); if (latest.revision > revision) { revision = latest.revision; controller.enqueue(encoder.encode(`id: ${revision}\nevent: shelf\ndata: ${JSON.stringify({ session_id: raw, ...JSON.parse(latest.latest_shelf) as Shelf })}\n\n`)); } if (once) break; if (turn % 15 === 0) controller.enqueue(encoder.encode(': keepalive\n\n')); await new Promise(resolve => setTimeout(resolve, 1000)); } if (!cancelled) controller.close(); } catch (error) { if (!cancelled) controller.error(error); } }, cancel() { cancelled = true; } });
      return new Response(stream, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no' } });
    }
    if ((action === 'event' || action === 'preferences') && request.method === 'POST') {
      await bounds(env, session); const input = await body(request), data = await loadData(env, url.origin); let feedback: FeedbackRow | undefined;
      if (action === 'preferences') session.genre = validateGenre(input.genre, data);
      else {
        if (typeof input.impression_id !== 'string' || !['click', 'save', 'rating'].includes(String(input.event))) fail(422, 'Valid impression_id and event required');
        const rating = input.rating === undefined ? null : input.rating; if ((input.event === 'rating') !== (rating !== null) || rating !== null && (typeof rating !== 'number' || !Number.isInteger(rating) || rating < 1 || rating > 5)) fail(422, 'Rating must be an integer from 1 to 5 only for rating events');
        const impression = await env.DB.prepare('SELECT * FROM stacks_impressions WHERE id=? AND session_hash=?').bind(input.impression_id, session.session_hash).first<ImpressionRow>(); if (!impression) fail(404, 'Impression not found in this session');
        const existing = await env.DB.prepare('SELECT id FROM stacks_feedback WHERE impression_id=? AND event=?').bind(impression.id, input.event).first(); if (existing) fail(409, 'Feedback already recorded');
        feedback = { id: crypto.randomUUID(), impression_id: impression.id, event: String(input.event), rating, feedback_at: new Date().toISOString() };
        session.seen = JSON.stringify([...new Set([...array<string>(session.seen), impression.item_id])]);
        if (input.event !== 'rating' || Number(rating) >= 4) session.history = JSON.stringify([...new Set([...array<string>(session.history), impression.item_id])]);
      }
      const generated = makeShelf(data, session); await writeShelf(env, session, generated, false, feedback); return json({ session_id: raw, ...generated.shelf });
    }
  }
  if (parts[0] === 'v1' && parts[1] === 'explain' && parts[2]) { const session = await getSession(env, url.searchParams.get('session_id') || ''); const row = await env.DB.prepare('SELECT * FROM stacks_impressions WHERE id=? AND session_hash=?').bind(parts[2], session.session_hash).first<ImpressionRow>(); if (!row) fail(404, 'Impression not found'); return json({ impression_id: row.id, item_id: row.item_id, position: row.position, propensity: row.propensity, impression_at: row.impression_at, trace: JSON.parse(row.trace) as unknown, statement: STATEMENT }); }
  if (path === '/v1/experiment/assign') { const session = await getSession(env, url.searchParams.get('session_id') || ''); return json({ experiment_id: EXPERIMENT, arm: session.arm, assignment_probability: .5, statement: STATEMENT }); }
  if (parts[0] === 'v1' && parts[1] === 'similar' && parts[2]) { const data = await loadData(env, url.origin), index = Number(parts[2]) - 1; if (!Number.isInteger(index) || index < 0 || index >= data.books.length) fail(404, 'Item not found'); const rows: { index: number; score: number }[] = []; for (let i = data.indptr[index]; i < data.indptr[index + 1]; i++) rows.push({ index: data.indices[i], score: data.cosine[i] }); return json({ items: rows.sort((a, b) => b.score - a.score || a.index - b.index).slice(0, 20).map(row => ({ ...data.books[row.index], score: row.score })), backend: 'evaluated-sparse-item-cosine', exact: false, detail: 'Stored top-neighbor index used identically by evaluation and serving.', statement: STATEMENT }); }
  if (path.startsWith('/v1/registry/') || path === '/v1/log') { requireAdmin(request, env); fail(501, 'Use the reference Python administrative API for this operation'); }
  return fail(404, 'Endpoint not found');
}
export async function handleRequest(request: Request, env: Env): Promise<Response> {
  const origins = (env.CORS_ORIGINS || '').split(',').map(origin => origin.trim()), origin = request.headers.get('Origin');
  const cors: Record<string, string> = { 'Vary': 'Origin', 'Access-Control-Allow-Methods': 'GET,POST,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type,X-Admin-Token,Last-Event-ID' };
  if (origin && origins.includes(origin)) cors['Access-Control-Allow-Origin'] = origin;
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: cors });
  let response: Response;
  try { response = await route(request, env); } catch (error) { response = error instanceof HttpError ? json({ detail: error.message }, error.status) : json({ detail: 'Service could not complete the request; verify artifacts and database initialization' }, 503); }
  const headers = new Headers(response.headers); for (const [name, value] of Object.entries(cors)) headers.set(name, value);
  return new Response(response.body, { status: response.status, headers });
}
export default { fetch: handleRequest };

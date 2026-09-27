import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import { build } from '../../web/node_modules/esbuild/lib/main.js';

const temporary = await mkdtemp(path.join(tmpdir(), 'stacks-edge-test-'));
const output = path.join(temporary, 'worker.mjs');
await build({ entryPoints: [new URL('./worker.ts', import.meta.url).pathname.replace(/^\/(?=[A-Z]:)/, '')], outfile: output, bundle: true, format: 'esm', platform: 'neutral', target: 'es2022' });
const { handleRequest, evaluateLoggedRows, scoreArtifacts } = await import(pathToFileURL(output).href);
const database = new DatabaseSync(':memory:');
database.exec(await readFile(new URL('./schema.sql', import.meta.url), 'utf8'));
let firstQueries = 0;
class Statement {
  constructor(sql, values = []) { this.sql = sql; this.values = values; }
  bind(...values) { return new Statement(this.sql, values); }
  async first() { firstQueries++; return database.prepare(this.sql).get(...this.values) || null; }
  async all() { return { results: database.prepare(this.sql).all(...this.values), meta: {} }; }
  async run() { const result = database.prepare(this.sql).run(...this.values); return { results: [], meta: { changes: Number(result.changes) } }; }
}
const db = {
  prepare(sql) { return new Statement(sql); },
  async batch(statements) { database.exec('BEGIN IMMEDIATE'); try { const result = []; for (const statement of statements) result.push(await statement.run()); database.exec('COMMIT'); return result; } catch (error) { database.exec('ROLLBACK'); throw error; } },
};
const books = Array.from({ length: 40 }, (_, index) => ({ id: index + 1, title: `Book ${index + 1}`, author: `Author ${index}`, genre: index % 2 ? 'Fiction' : 'Science', tags: [index % 2 ? 'fiction' : 'science'], popularity: 40 - index, year: 2000 }));
const manifest = { schema_version: '1', artifact_version: 'test-artifact-v1', catalog_size: 40, factors: 2, alpha: 20, regularization: .1 };
const profiles = [{ id: '1', positive_indices: [0, 1, 2], seen_indices: [0, 1, 2, 3], user_factor: [.5, .25] }];
const factors = Float32Array.from(books.flatMap((_, i) => [i / 40, (40 - i) / 40]));
const popularity = Float32Array.from(books.map(book => book.popularity));
const cosine = Float32Array.from(books.map(() => .5));
const indices = Uint32Array.from(books.map((_, i) => (i + 1) % 40));
const indptr = Uint32Array.from(Array.from({ length: 41 }, (_, i) => i));
const assets = new Map([
  ['edge-manifest.json', JSON.stringify(manifest)], ['catalog.json', JSON.stringify(books)], ['edge-profiles.json', JSON.stringify(profiles)],
  ['item-factors.f32', factors], ['popularity.f32', popularity], ['cosine-data.f32', cosine], ['cosine-indices.u32', indices], ['cosine-indptr.u32', indptr],
]);
const env = { DB: db, SESSION_HASH_KEY: 'test-persistent-secret-12345', ASSETS: { async fetch(request) { const value = assets.get(new URL(request.url).pathname.split('/').at(-1)); return value === undefined ? new Response('missing', { status: 404 }) : new Response(value); } } };
const request = (route, body) => handleRequest(new Request(`https://stacks.test${route}`, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }), env);

test('artifact scoring uses exact saved reader factors and published blend weights', () => {
  const data = { manifest, books, profiles: new Map([['1', profiles[0]]]), factors, popularity, cosine, indices, indptr };
  const result = scoreArtifacts(data, ['1', '2', '3'], '1');
  assert.equal(result.mode, 'evaluated-reader-exact');
  for (const row of result.rows) assert.ok(Math.abs(row.scores.als - (factors[row.index * 2] * .5 + factors[row.index * 2 + 1] * .25)) < 1e-12);
  assert.equal(scoreArtifacts(data, ['1', '2', '3', '5'], '1').mode, 'frozen-item-factor-session-fold-in');
  assert.equal(scoreArtifacts(data, [], null).mode, 'cold-popularity');
});

test('D1 session, persisted propensities, SSE resume and owned events round trip', async () => {
  const response = await request('/v1/session', { reader_id: '1' });
  assert.equal(response.status, 201, await response.clone().text());
  const shelf = await response.json(), raw = shelf.session_id;
  assert.equal(shelf.items.length, 10);
  assert.equal(shelf.scoring_mode, 'evaluated-reader-exact');
  assert.ok(shelf.items.every(item => !['1', '2', '3', '4'].includes(item.id)));
  assert.ok(shelf.items.slice(0, -1).every(item => item.propensity === 1));
  assert.equal(shelf.items.at(-1).propensity, .05);
  const resume = await request(`/v1/session/${raw}`);
  assert.equal((await resume.json()).revision, 1);
  const stream = await request(`/v1/session/${raw}/stream?once=true`);
  assert.match(await stream.text(), /event: shelf/);
  assert.equal(database.prepare('SELECT COUNT(*) n FROM stacks_impressions').get().n, 10);
  const clicked = shelf.items[0];
  const nextResponse = await request(`/v1/session/${raw}/event`, { impression_id: clicked.impression_id, event: 'click' });
  assert.equal(nextResponse.status, 200, await nextResponse.clone().text());
  const next = await nextResponse.json(); assert.equal(next.revision, 2); assert.equal(next.scoring_mode, 'frozen-item-factor-session-fold-in');
  assert.ok(next.items.every(item => item.id !== clicked.id));
  assert.equal(database.prepare('SELECT COUNT(*) n FROM stacks_feedback').get().n, 1);
  assert.ok(!database.prepare('SELECT latest_shelf FROM stacks_sessions').get().latest_shelf.includes(raw));
  assert.equal((await request(`/v1/session/${raw}/event`, { impression_id: clicked.impression_id, event: 'click' })).status, 409);
  const other = await (await request('/v1/session', {})).json();
  assert.equal((await request(`/v1/session/${other.session_id}/event`, { impression_id: clicked.impression_id, event: 'click' })).status, 404);
  const changed = await (await request(`/v1/session/${raw}/preferences`, { genre: 'Fiction' })).json();
  assert.ok(changed.items.every(item => item.genre === 'Fiction'));
  const logs = await (await request(`/v1/session/${raw}/logs`)).json();
  assert.ok(logs.impressions.every(row => !Object.hasOwn(row, 'session_hash')));
  assert.equal((await (await request(`/v1/session/${raw}/ope`)).json()).status, 'awaiting_mature_feedback');
});

test('SSE closes and resumes within the D1 Free invocation query budget', async context => {
  const shelf = await (await request('/v1/session', {})).json();
  const originalTimeout = globalThis.setTimeout;
  context.mock.method(globalThis, 'setTimeout', (callback, _delay, ...args) => originalTimeout(callback, 0, ...args));
  const before = firstQueries;
  const stream = await request(`/v1/session/${shelf.session_id}/stream`);
  const payload = await stream.text();
  assert.match(payload, /event: shelf/);
  assert.ok(firstQueries - before <= 50, 'The initial lookup plus every poll must fit one Free D1 invocation');
  const resumed = await handleRequest(new Request(`https://stacks.test/v1/session/${shelf.session_id}/stream?once=true`, { headers: { 'Last-Event-ID': String(shelf.revision) } }), env);
  assert.doesNotMatch(await resumed.text(), /event: shelf/, 'Reconnect must not emit the same revision twice');
});

test('own-log OPE enforces immutable horizon and support with conservative intervals', () => {
  const now = Date.now(), pool = ['1', '2'];
  const trace = { exploration: { target_policy: 'final-slot-80-best-20-uniform-v1', candidate_pool: pool, target_probabilities: [.9, .1], reward_model: [.05, .05], reward_horizon_seconds: 60 } };
  const base = { id: 'a', session_hash: 'hash', item_id: '1', policy: 'uniform', propensity: .5, trace: JSON.stringify(trace), impression_at: new Date(now - 70000).toISOString() };
  const feedback = [{ id: 'f', impression_id: 'a', event: 'click', rating: null, feedback_at: new Date(now - 20000).toISOString() }];
  const result = evaluateLoggedRows([base], feedback, now);
  assert.equal(result.status, 'measured'); assert.equal(result.matured, 1); assert.equal(result.full_slate_identified, false);
  assert.equal(result.estimates.IPS.mean, 1.8); assert.equal(result.estimates.SNIPS.mean, 1); assert.equal(result.estimates.DM.mean, .05);
  assert.ok(result.estimates.IPS.low < 0 && result.estimates.IPS.high > 1);
  const pending = evaluateLoggedRows([{ ...base, impression_at: new Date(now - 10000).toISOString() }], feedback, now);
  assert.equal(pending.pending, 1); assert.equal(pending.matured, 0);
  const late = evaluateLoggedRows([base], [{ ...feedback[0], feedback_at: new Date(now - 1000).toISOString() }], now);
  assert.equal(late.late_feedback_ignored, 1); assert.equal(late.estimates.IPS.mean, 0);
  const invalid = evaluateLoggedRows([{ ...base, propensity: .2 }], feedback, now);
  assert.equal(invalid.invalid_excluded, 1); assert.equal(invalid.matured, 0);
  assert.equal(evaluateLoggedRows([{ ...base, policy: 'deterministic' }], feedback, now).matured, 0);
});

test('chunked public writes stop reading at the body size limit', async () => {
  let sent = 0;
  const stream = new ReadableStream({ pull(controller) { sent++; controller.enqueue(new Uint8Array(20000)); } });
  const response = await handleRequest(new Request('https://stacks.test/v1/session', { method: 'POST', body: stream, duplex: 'half' }), env);
  assert.equal(response.status, 413);
  assert.ok(sent <= 3, 'The reader must cancel rather than buffer an unbounded upload');
});

test('all exported public readers match evaluated artifact scores and ranking', async () => {
  const directory = new URL('../../web/public/data/', import.meta.url);
  const actualManifest = JSON.parse(await readFile(new URL('edge-manifest.json', directory), 'utf8'));
  const actualBooks = JSON.parse(await readFile(new URL('catalog.json', directory), 'utf8'));
  const actualProfiles = JSON.parse(await readFile(new URL('edge-profiles.json', directory), 'utf8'));
  const fixtures = JSON.parse(await readFile(new URL('edge-parity.json', directory), 'utf8'));
  const binary = async (name, ArrayType) => { const bytes = await readFile(new URL(name, directory)); return new ArrayType(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)); };
  const actual = { manifest: actualManifest, books: actualBooks, profiles: new Map(actualProfiles.map(profile => [String(profile.id), profile])), popularity: await binary('popularity.f32', Float32Array), factors: await binary('item-factors.f32', Float32Array), cosine: await binary('cosine-data.f32', Float32Array), indices: await binary('cosine-indices.u32', Uint32Array), indptr: await binary('cosine-indptr.u32', Uint32Array) };
  assert.equal(fixtures.readers.length, actualProfiles.length);
  assert.equal(fixtures.readers.length, 24);
  for (const fixture of fixtures.readers) {
    const profile = actual.profiles.get(String(fixture.id));
    const scores = scoreArtifacts(actual, profile.positive_indices.map(i => String(i + 1)), String(fixture.id));
    const seen = new Set(profile.seen_indices);
    for (const model of ['popularity', 'item_cosine', 'als', 'blend']) {
      const expected = fixture.models[model];
      const ranked = scores.rows.filter(row => !seen.has(row.index)).sort((a, b) => b.scores[model] - a.scores[model] || a.index - b.index).slice(0, expected.length);
      assert.deepEqual(ranked.map(row => Number(row.id)), expected.map(row => row.id), `Reader ${fixture.id}, ${model} ranking`);
      for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(ranked[i].scores[model] - expected[i].score) <= fixtures.absolute_score_tolerance, `Reader ${fixture.id}, ${model}, item ${expected[i].id} score`);
    }
  }
});

test.after(async () => { database.close(); await rm(temporary, { recursive: true, force: true }); });

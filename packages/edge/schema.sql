CREATE TABLE IF NOT EXISTS stacks_sessions (
  session_hash TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  arm TEXT NOT NULL,
  reader_id TEXT,
  history TEXT NOT NULL,
  seen TEXT NOT NULL,
  genre TEXT NOT NULL DEFAULT 'All books',
  revision INTEGER NOT NULL,
  operation_id TEXT NOT NULL,
  latest_shelf TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stacks_impressions (
  id TEXT PRIMARY KEY,
  session_hash TEXT NOT NULL REFERENCES stacks_sessions(session_hash),
  item_id TEXT NOT NULL,
  position INTEGER NOT NULL CHECK(position > 0),
  propensity REAL NOT NULL CHECK(propensity > 0 AND propensity <= 1),
  arm TEXT NOT NULL,
  model_version TEXT NOT NULL,
  impression_at TEXT NOT NULL,
  policy TEXT NOT NULL,
  candidate_pool TEXT NOT NULL,
  trace TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS stacks_impressions_session ON stacks_impressions(session_hash,impression_at);
CREATE INDEX IF NOT EXISTS stacks_impressions_policy ON stacks_impressions(policy,impression_at);
CREATE TABLE IF NOT EXISTS stacks_feedback (
  id TEXT PRIMARY KEY,
  impression_id TEXT NOT NULL REFERENCES stacks_impressions(id),
  event TEXT NOT NULL CHECK(event IN ('click','save','rating')),
  rating INTEGER CHECK(rating IS NULL OR rating BETWEEN 1 AND 5),
  feedback_at TEXT NOT NULL,
  UNIQUE(impression_id,event)
);
CREATE TABLE IF NOT EXISTS stacks_rate_limits (
  bucket TEXT PRIMARY KEY,
  count INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);

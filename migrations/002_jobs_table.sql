BEGIN;

SET LOCAL search_path TO public, extensions;

CREATE SCHEMA IF NOT EXISTS concept_explainer;
CREATE TABLE IF NOT EXISTS concept_explainer.jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id TEXT UNIQUE NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued',
  payload JSONB NOT NULL,
  attempts INT NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_status_created_at_idx
  ON concept_explainer.jobs (status, created_at);

COMMIT;

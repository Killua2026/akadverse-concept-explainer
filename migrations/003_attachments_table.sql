BEGIN;
SET LOCAL search_path TO public, extensions;
CREATE SCHEMA IF NOT EXISTS concept_explainer;
CREATE TABLE IF NOT EXISTS concept_explainer.attachments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  size_bytes INT NOT NULL,
  content BYTEA NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS attachments_expires_at_idx
  ON concept_explainer.attachments (expires_at);
COMMIT;

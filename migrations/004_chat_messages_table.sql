BEGIN;
SET LOCAL search_path TO public, extensions;
CREATE SCHEMA IF NOT EXISTS concept_explainer;
CREATE TABLE IF NOT EXISTS concept_explainer.chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  sources_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_created_idx
  ON concept_explainer.chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS chat_messages_created_at_idx
  ON concept_explainer.chat_messages (created_at);
COMMIT;

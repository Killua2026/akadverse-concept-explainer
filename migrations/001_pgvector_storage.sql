/*
Run this file as a single transaction (psql -f, or paste the whole block into the Supabase SQL editor). Do not run the statements one at a time: SET LOCAL search_path will not stick, and CREATE TABLE will fail on vector(768).
*/
BEGIN;

SET LOCAL search_path TO public, extensions;

CREATE SCHEMA IF NOT EXISTS concept_explainer;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS concept_explainer.chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(768) NOT NULL,
  metadata JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
  ON concept_explainer.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
  ON concept_explainer.chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_course_offering_id_idx
  ON concept_explainer.chunks ((metadata->>'course_offering_id'));

COMMIT;

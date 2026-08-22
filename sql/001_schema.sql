-- docs-rag schema.
-- Mounted into /docker-entrypoint-initdb.d, so it runs once when the pgvector
-- container initialises an empty data directory.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  id          SERIAL PRIMARY KEY,
  source      TEXT NOT NULL UNIQUE,   -- filename or EDGAR URL
  title       TEXT,
  ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id          SERIAL PRIMARY KEY,
  document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  content     TEXT NOT NULL,
  embedding   VECTOR(384),            -- must match EMBEDDING_DIM
  UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- The IVFFlat index is NOT created here on purpose.
--
-- IVFFlat is a *trained* index: it k-means the existing rows into `lists`
-- centroids and only probes a few of them at query time. Building it against an
-- empty table produces useless centroids, and every row inserted afterwards is
-- assigned to whatever partitioning that empty build invented. So it is built
-- after ingestion, by `python -m app.cli build-index`, which also sizes `lists`
-- from the actual row count (pgvector's rule of thumb: rows/1000, min 1).

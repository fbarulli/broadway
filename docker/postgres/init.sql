CREATE TABLE IF NOT EXISTS etl_file_manifest (
    filename TEXT PRIMARY KEY,
    row_count INTEGER NOT NULL,
    loaded_at TIMESTAMPTZ DEFAULT now()
);

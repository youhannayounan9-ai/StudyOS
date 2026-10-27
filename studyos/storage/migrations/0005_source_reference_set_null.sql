-- StudyOS Migration 0005: Correct exam_question_sources foreign key to ON DELETE SET NULL
-- Safely rebuilds exam_question_sources so deleting source chunks or documents
-- retains historical citation provenance and snapshot metadata rather than cascading deletion.

CREATE TABLE new_exam_question_sources (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES exam_questions(id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    evidence_marker TEXT NOT NULL,
    original_chunk_id TEXT NOT NULL DEFAULT '',
    snapshot_filename TEXT NOT NULL DEFAULT '',
    snapshot_location_type TEXT,
    snapshot_location_index INTEGER,
    snapshot_location_title TEXT,
    snapshot_excerpt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO new_exam_question_sources (
    id, question_id, chunk_id, evidence_marker,
    original_chunk_id, snapshot_filename, snapshot_location_type,
    snapshot_location_index, snapshot_location_title, snapshot_excerpt,
    created_at
)
SELECT
    id, question_id, chunk_id, evidence_marker,
    original_chunk_id, snapshot_filename, snapshot_location_type,
    snapshot_location_index, snapshot_location_title, snapshot_excerpt,
    created_at
FROM exam_question_sources;

DROP TABLE exam_question_sources;

ALTER TABLE new_exam_question_sources RENAME TO exam_question_sources;

CREATE INDEX IF NOT EXISTS idx_exam_question_sources_question_id ON exam_question_sources(question_id);
CREATE INDEX IF NOT EXISTS idx_exam_question_sources_chunk_id ON exam_question_sources(chunk_id);

-- StudyOS Migration 0004: Grading States and Historical Citation Snapshots
-- Upgrades assessment engine schema to support:
-- 1. Explicit attempt statuses ('pending_review') and provisional subtotals.
-- 2. Explicit question attempt statuses ('graded', 'pending_manual_review') with nullable scores.
-- 3. Historical source snapshots on exam_question_sources preserving provenance when chunks/docs are deleted.

-- Step 1: Add provisional_score to exam_attempts
ALTER TABLE exam_attempts ADD COLUMN provisional_score REAL;

-- Step 2: Add status to question_attempts with default 'graded'
ALTER TABLE question_attempts ADD COLUMN status TEXT NOT NULL DEFAULT 'graded';

-- Step 3: Add snapshot columns to exam_question_sources
ALTER TABLE exam_question_sources ADD COLUMN original_chunk_id TEXT NOT NULL DEFAULT '';
ALTER TABLE exam_question_sources ADD COLUMN snapshot_filename TEXT NOT NULL DEFAULT '';
ALTER TABLE exam_question_sources ADD COLUMN snapshot_location_type TEXT;
ALTER TABLE exam_question_sources ADD COLUMN snapshot_location_index INTEGER;
ALTER TABLE exam_question_sources ADD COLUMN snapshot_location_title TEXT;
ALTER TABLE exam_question_sources ADD COLUMN snapshot_excerpt TEXT;

-- Step 4: Backfill historical snapshots for any existing exam_question_sources rows
UPDATE exam_question_sources
SET
    original_chunk_id = COALESCE(NULLIF(original_chunk_id, ''), chunk_id, 'deleted_chunk'),
    snapshot_filename = COALESCE(
        NULLIF(snapshot_filename, ''),
        (
            SELECT d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = exam_question_sources.chunk_id
        ),
        'unavailable_document'
    ),
    snapshot_location_type = COALESCE(
        snapshot_location_type,
        (
            SELECT c.location_type
            FROM chunks c
            WHERE c.id = exam_question_sources.chunk_id
        )
    ),
    snapshot_location_index = COALESCE(
        snapshot_location_index,
        (
            SELECT c.location_index
            FROM chunks c
            WHERE c.id = exam_question_sources.chunk_id
        )
    ),
    snapshot_location_title = COALESCE(
        snapshot_location_title,
        (
            SELECT c.location_title
            FROM chunks c
            WHERE c.id = exam_question_sources.chunk_id
        )
    ),
    snapshot_excerpt = COALESCE(
        snapshot_excerpt,
        (
            SELECT c.content
            FROM chunks c
            WHERE c.id = exam_question_sources.chunk_id
        )
    );

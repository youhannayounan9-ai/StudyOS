-- StudyOS Migration 0006: Persistent Learner Model, Concept Catalog & Spaced Review
-- Adds normalized tables for learner profile, course-scoped concept catalog,
-- question-concept associations, concept-chunk grounding, concept mastery tracking,
-- chronological learner events, and review task queue.

-- 1. Learner Profile
CREATE TABLE IF NOT EXISTS learner_profile (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed default local learner profile
INSERT OR IGNORE INTO learner_profile (id, name)
VALUES ('learner_default', 'Local Learner');

-- 2. Course-Scoped Concepts Catalog
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(course_id, normalized_name)
);

CREATE INDEX IF NOT EXISTS idx_concepts_course_id ON concepts(course_id);
CREATE INDEX IF NOT EXISTS idx_concepts_norm_name ON concepts(course_id, normalized_name);

-- 3. Question-to-Concept Junction
CREATE TABLE IF NOT EXISTS question_concepts (
    question_id TEXT NOT NULL REFERENCES exam_questions(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (question_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_question_concepts_concept_id ON question_concepts(concept_id);

-- 4. Concept-to-Source-Chunk Junction
CREATE TABLE IF NOT EXISTS concept_chunks (
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (concept_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_concept_chunks_chunk_id ON concept_chunks(chunk_id);

-- 5. Concept Mastery
CREATE TABLE IF NOT EXISTS concept_mastery (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learner_profile(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    mastery_score REAL NOT NULL DEFAULT 0.0,
    mastery_category TEXT NOT NULL DEFAULT 'unassessed',
    observation_count INTEGER NOT NULL DEFAULT 0,
    correct_count REAL NOT NULL DEFAULT 0.0,
    incorrect_count REAL NOT NULL DEFAULT 0.0,
    uncertainty REAL NOT NULL DEFAULT 0.2887,
    last_assessed_at TIMESTAMP,
    next_review_due TIMESTAMP,
    interval_days REAL NOT NULL DEFAULT 1.0,
    ease_factor REAL NOT NULL DEFAULT 2.5,
    repetition_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(learner_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_concept_mastery_course_id ON concept_mastery(course_id);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_learner_id ON concept_mastery(learner_id);
CREATE INDEX IF NOT EXISTS idx_concept_mastery_category ON concept_mastery(mastery_category);

-- 6. Chronological Learner Activity Events
CREATE TABLE IF NOT EXISTS learner_events (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learner_profile(id) ON DELETE CASCADE,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    concept_id TEXT REFERENCES concepts(id) ON DELETE SET NULL,
    event_type TEXT NOT NULL, -- 'exam_question_graded', 'review_completed', 'manual_grade_finalized'
    reference_id TEXT, -- e.g. attempt_id or review_task_id
    question_id TEXT REFERENCES exam_questions(id) ON DELETE SET NULL,
    score_achieved REAL,
    score_max REAL,
    is_correct INTEGER, -- 1 = correct, 0 = incorrect, NULL = not applicable / pending
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_learner_events_learner_id ON learner_events(learner_id);
CREATE INDEX IF NOT EXISTS idx_learner_events_course_id ON learner_events(course_id);
CREATE INDEX IF NOT EXISTS idx_learner_events_concept_id ON learner_events(concept_id);
CREATE INDEX IF NOT EXISTS idx_learner_events_ref_id ON learner_events(reference_id);
CREATE INDEX IF NOT EXISTS idx_learner_events_created_at ON learner_events(created_at);

-- 7. Spaced Review Tasks
CREATE TABLE IF NOT EXISTS review_tasks (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learner_profile(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    due_date TIMESTAMP NOT NULL,
    status TEXT NOT NULL DEFAULT 'scheduled', -- 'scheduled', 'completed', 'skipped'
    completed_at TIMESTAMP,
    rating TEXT, -- 'again', 'hard', 'good', 'easy'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_review_tasks_due_date ON review_tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_review_tasks_status ON review_tasks(status);
CREATE INDEX IF NOT EXISTS idx_review_tasks_concept_id ON review_tasks(concept_id);
CREATE INDEX IF NOT EXISTS idx_review_tasks_course_id ON review_tasks(course_id);

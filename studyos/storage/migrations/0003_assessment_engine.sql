-- StudyOS Migration 0003: Assessment Engine Schema
-- Defines normalized relational persistence for exams, questions, source references, attempts, and grading.

CREATE TABLE IF NOT EXISTS exams (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    topic TEXT,
    difficulty TEXT NOT NULL,
    question_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exams_course_id ON exams(course_id);

CREATE TABLE IF NOT EXISTS exam_questions (
    id TEXT PRIMARY KEY,
    exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    question_index INTEGER NOT NULL,
    question_type TEXT NOT NULL, -- 'mcq', 'true_false', 'short_answer'
    difficulty TEXT NOT NULL,
    question_text TEXT NOT NULL,
    options_json TEXT, -- JSON representation of choices: [{"id": "A", "text": "..."}]
    correct_answer TEXT NOT NULL,
    explanation TEXT NOT NULL,
    rubric_json TEXT, -- JSON representation of short-answer scoring criteria
    points REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exam_questions_exam_id ON exam_questions(exam_id);

CREATE TABLE IF NOT EXISTS exam_question_sources (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES exam_questions(id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES chunks(id) ON DELETE CASCADE,
    evidence_marker TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exam_question_sources_question_id ON exam_question_sources(question_id);
CREATE INDEX IF NOT EXISTS idx_exam_question_sources_chunk_id ON exam_question_sources(chunk_id);

CREATE TABLE IF NOT EXISTS exam_attempts (
    id TEXT PRIMARY KEY,
    exam_id TEXT NOT NULL REFERENCES exams(id) ON DELETE CASCADE,
    status TEXT NOT NULL, -- 'in_progress', 'completed'
    score REAL,
    max_score REAL,
    percentage REAL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exam_attempts_exam_id ON exam_attempts(exam_id);

CREATE TABLE IF NOT EXISTS question_attempts (
    id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES exam_attempts(id) ON DELETE CASCADE,
    question_id TEXT NOT NULL REFERENCES exam_questions(id) ON DELETE CASCADE,
    submitted_answer TEXT,
    is_correct INTEGER, -- 1 = correct, 0 = incorrect, NULL = manual review / pending
    score REAL,
    max_score REAL,
    feedback TEXT,
    graded_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_question_attempts_attempt_id ON question_attempts(attempt_id);

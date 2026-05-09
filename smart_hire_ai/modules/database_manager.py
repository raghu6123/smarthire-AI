"""
SmartHire AI – Database Manager
SQLite database operations for candidates, jobs, and predictions.
"""

import sqlite3
import os
from datetime import datetime
from utils.constants import DATABASE_PATH, DATABASE_DIR


class DatabaseManager:
    """Manages all SQLite database operations for SmartHire AI."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._create_tables()

    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Create all required tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT DEFAULT '',
                skills TEXT DEFAULT '',
                education TEXT DEFAULT '',
                experience INTEGER DEFAULT 0,
                similarity_score REAL DEFAULT 0.0,
                suitability_score REAL DEFAULT 0.0,
                resume_text TEXT DEFAULT '',
                resume_path TEXT DEFAULT '',
                department TEXT DEFAULT '',
                gender TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                required_skills TEXT DEFAULT '',
                description TEXT DEFAULT '',
                department TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                prediction INTEGER DEFAULT 0,
                probability REAL DEFAULT 0.0,
                model_used TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            )
        """)

        conn.commit()
        conn.close()

    # ===== CANDIDATE OPERATIONS =====

    def add_candidate(self, name, email="", skills="", education="",
                      experience=0, resume_text="", resume_path="",
                      department="", gender=""):
        """Add a new candidate to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO candidates (name, email, skills, education, experience,
                                    resume_text, resume_path, department, gender)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, email, skills, education, experience,
              resume_text, resume_path, department, gender))
        candidate_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return candidate_id

    def get_all_candidates(self):
        """Get all candidates as a list of dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates ORDER BY created_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_candidate_by_id(self, candidate_id):
        """Get a single candidate by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_candidate_scores(self, candidate_id, similarity_score=None,
                                 suitability_score=None):
        """Update similarity and/or suitability scores for a candidate."""
        conn = self._get_connection()
        cursor = conn.cursor()
        if similarity_score is not None:
            cursor.execute("UPDATE candidates SET similarity_score = ? WHERE id = ?",
                           (similarity_score, candidate_id))
        if suitability_score is not None:
            cursor.execute("UPDATE candidates SET suitability_score = ? WHERE id = ?",
                           (suitability_score, candidate_id))
        conn.commit()
        conn.close()

    def update_candidate_notes(self, candidate_id, notes):
        """Update recruiter notes for a candidate."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE candidates SET notes = ? WHERE id = ?",
                       (notes, candidate_id))
        conn.commit()
        conn.close()

    def search_candidates(self, query):
        """Search candidates by name or skills."""
        conn = self._get_connection()
        cursor = conn.cursor()
        search = f"%{query}%"
        cursor.execute("""
            SELECT * FROM candidates
            WHERE name LIKE ? OR skills LIKE ? OR email LIKE ?
            ORDER BY similarity_score DESC
        """, (search, search, search))
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_candidate_count(self):
        """Get total number of candidates."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM candidates")
        count = cursor.fetchone()["count"]
        conn.close()
        return count

    def delete_candidate(self, candidate_id):
        """Delete a candidate."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE candidate_id = ?", (candidate_id,))
        cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
        conn.commit()
        conn.close()

    # ===== JOB OPERATIONS =====

    def add_job(self, role, required_skills="", description="", department=""):
        """Add a new job to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (role, required_skills, description, department)
            VALUES (?, ?, ?, ?)
        """, (role, required_skills, description, department))
        job_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return job_id

    def get_all_jobs(self):
        """Get all jobs."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_job_by_id(self, job_id):
        """Get a single job by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ===== PREDICTION OPERATIONS =====

    def add_prediction(self, candidate_id, prediction, probability, model_used=""):
        """Add a prediction result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (candidate_id, prediction, probability, model_used)
            VALUES (?, ?, ?, ?)
        """, (candidate_id, prediction, probability, model_used))
        conn.commit()
        conn.close()

    def get_predictions(self):
        """Get all predictions with candidate info."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*, c.name, c.skills, c.experience
            FROM predictions p
            JOIN candidates c ON p.candidate_id = c.id
            ORDER BY p.probability DESC
        """)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    # ===== ANALYTICS QUERIES =====

    def get_avg_similarity_score(self):
        """Get average similarity score."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT AVG(similarity_score) as avg FROM candidates WHERE similarity_score > 0")
        row = cursor.fetchone()
        conn.close()
        return row["avg"] if row and row["avg"] else 0.0

    def get_skill_distribution(self):
        """Get skill frequency distribution."""
        candidates = self.get_all_candidates()
        skill_counts = {}
        for c in candidates:
            if c.get("skills"):
                for skill in c["skills"].split(","):
                    skill = skill.strip().lower()
                    if skill:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
        return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True))

    def get_experience_distribution(self):
        """Get experience level distribution."""
        candidates = self.get_all_candidates()
        dist = {"0-2 yrs": 0, "2-5 yrs": 0, "5-10 yrs": 0, "10+ yrs": 0}
        for c in candidates:
            exp = c.get("experience", 0) or 0
            if exp <= 2:
                dist["0-2 yrs"] += 1
            elif exp <= 5:
                dist["2-5 yrs"] += 1
            elif exp <= 10:
                dist["5-10 yrs"] += 1
            else:
                dist["10+ yrs"] += 1
        return dist

    def get_department_distribution(self):
        """Get department-wise candidate distribution."""
        candidates = self.get_all_candidates()
        dept_counts = {}
        for c in candidates:
            dept = c.get("department", "Unassigned") or "Unassigned"
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        return dept_counts

    def clear_all_data(self):
        """Clear all data from all tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM candidates")
        cursor.execute("DELETE FROM jobs")
        conn.commit()
        conn.close()

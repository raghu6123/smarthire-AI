"""
SmartHire AI – Constants & Configuration
Centralized constants for colors, skills, scoring weights, and paths.
"""

import os

# =====================================================
# PATH CONSTANTS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "recruitment.db")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESUMES_DIR = os.path.join(DATA_DIR, "resumes")
JOBS_DIR = os.path.join(DATA_DIR, "jobs")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MODEL_PATH = os.path.join(MODELS_DIR, "suitability_model.pkl")

# =====================================================
# UI COLOR PALETTE (Dark AI SaaS Theme)
# =====================================================
COLORS = {
    "bg_primary": "#0a0e1a",         # Deep dark navy
    "bg_secondary": "#111827",       # Dark card background
    "bg_card": "rgba(17, 24, 39, 0.7)",  # Glassmorphism card
    "accent_primary": "#6366f1",     # Indigo
    "accent_secondary": "#8b5cf6",   # Purple
    "accent_tertiary": "#06b6d4",    # Cyan
    "gradient_start": "#6366f1",     # Indigo
    "gradient_end": "#a855f7",       # Purple
    "text_primary": "#f1f5f9",       # Near white
    "text_secondary": "#94a3b8",     # Muted slate
    "text_muted": "#64748b",         # Dim slate
    "success": "#10b981",            # Emerald green
    "warning": "#f59e0b",            # Amber
    "danger": "#ef4444",             # Red
    "info": "#3b82f6",               # Blue
    "border": "rgba(99, 102, 241, 0.2)",  # Subtle indigo border
    "shadow": "rgba(0, 0, 0, 0.3)",
}

# Plotly-compatible color sequence
PLOTLY_COLORS = [
    "#6366f1", "#8b5cf6", "#a855f7", "#06b6d4",
    "#10b981", "#f59e0b", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#84cc16",
]

PLOTLY_TEMPLATE = "plotly_dark"

# =====================================================
# CUSTOM SKILL DICTIONARY
# =====================================================
SKILL_DICTIONARY = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "swift", "kotlin", "php", "scala", "r", "matlab", "perl",
    
    # Web Technologies
    "html", "css", "react", "angular", "vue", "node.js", "express",
    "django", "flask", "fastapi", "spring", "nextjs", "gatsby",
    
    # Data Science & ML
    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "data engineering", "nlp",
    "natural language processing", "computer vision", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "seaborn", "plotly", "statistics", "data visualization",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "firebase", "oracle", "sqlite",
    
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "jenkins", "ci/cd", "devops", "linux", "git", "github",
    "gitlab", "ansible", "nginx",
    
    # Other Technologies
    "api", "rest", "graphql", "microservices", "agile", "scrum",
    "jira", "figma", "tableau", "power bi", "excel", "spark",
    "hadoop", "kafka", "airflow", "blockchain", "cybersecurity",
    
    # Soft Skills
    "leadership", "communication", "teamwork", "problem solving",
    "critical thinking", "project management", "time management",
}

# Skill categories for analytics
SKILL_CATEGORIES = {
    "Programming": {"python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "rust", "swift", "kotlin", "php", "scala", "r"},
    "Web Development": {"html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask", "nextjs"},
    "Data Science & ML": {"machine learning", "deep learning", "data science", "data analysis", "nlp", "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"},
    "Cloud & DevOps": {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "devops", "linux"},
    "Databases": {"sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle", "sqlite"},
    "Soft Skills": {"leadership", "communication", "teamwork", "problem solving", "project management"},
}

# =====================================================
# EDUCATION SCORING
# =====================================================
EDUCATION_LEVELS = {
    "phd": 5,
    "doctorate": 5,
    "master": 4,
    "mba": 4,
    "m.tech": 4,
    "m.s.": 4,
    "bachelor": 3,
    "b.tech": 3,
    "b.e.": 3,
    "b.s.": 3,
    "b.sc": 3,
    "associate": 2,
    "diploma": 1,
    "high school": 0,
}

EDUCATION_MAX_SCORE = 5

# =====================================================
# EXPERIENCE SCORING
# =====================================================
EXPERIENCE_WEIGHTS = {
    "entry": {"min_years": 0, "max_years": 2, "score": 1},
    "junior": {"min_years": 2, "max_years": 4, "score": 2},
    "mid": {"min_years": 4, "max_years": 7, "score": 3},
    "senior": {"min_years": 7, "max_years": 12, "score": 4},
    "lead": {"min_years": 12, "max_years": 100, "score": 5},
}

EXPERIENCE_MAX_SCORE = 5

# =====================================================
# RANKING WEIGHTS
# =====================================================
RANKING_WEIGHTS = {
    "similarity_score": 0.40,
    "skills_match": 0.25,
    "experience_score": 0.20,
    "education_score": 0.15,
}

# =====================================================
# ML MODEL CONFIGURATION
# =====================================================
ML_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_synthetic_samples": 500,
    "models": {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
    },
}

# =====================================================
# BIAS DETECTION THRESHOLDS
# =====================================================
BIAS_THRESHOLDS = {
    "gender_skew_warning": 0.3,    # Warn if gender ratio below 30%
    "experience_skew_warning": 0.2, # Warn if experience group < 20%
    "skill_bias_warning": 0.15,     # Warn if skill group < 15%
}

# =====================================================
# PAGE CONFIGURATION
# =====================================================
PAGES = {
    "Dashboard": "dashboard",
    "Upload Resumes": "upload",
    "Job Matching": "matching",
    "Candidate Ranking": "ranking",
    "Suitability Prediction": "prediction",
    "Bias Detection": "bias",
    "Analytics Dashboard": "analytics",
    "Report Generator": "reports",
}

# =====================================================
# SAMPLE DATA (for demo/initial state)
# =====================================================
SAMPLE_DEPARTMENTS = [
    "Engineering", "Data Science", "Product Management",
    "Design", "Marketing", "Sales", "HR", "Finance",
]

SAMPLE_JOB_ROLES = [
    "Software Engineer", "Data Scientist", "ML Engineer",
    "Frontend Developer", "Backend Developer", "DevOps Engineer",
    "Product Manager", "UX Designer", "Data Analyst",
    "Full Stack Developer",
]

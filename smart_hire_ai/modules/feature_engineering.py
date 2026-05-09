"""
SmartHire AI – Feature Engineering
Generate TF-IDF vectors, skill matrices, and scoring features for ML models.
"""

# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.constants import EDUCATION_LEVELS, EDUCATION_MAX_SCORE, EXPERIENCE_MAX_SCORE


class FeatureEngineer:
    """Generates features from candidate data for similarity and ML models."""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        self._is_fitted = False

    def fit_tfidf(self, documents):
        """Fit the TF-IDF vectorizer on a list of documents."""
        if not documents:
            return None
        clean_docs = [doc if doc else "" for doc in documents]
        self.tfidf_vectorizer.fit(clean_docs)
        self._is_fitted = True
        return self.tfidf_vectorizer

    def transform_tfidf(self, documents):
        """Transform documents into TF-IDF vectors. Fits if not already fitted."""
        if not documents:
            return None
        clean_docs = [doc if doc else "" for doc in documents]
        if not self._is_fitted:
            self._is_fitted = True
            return self.tfidf_vectorizer.fit_transform(clean_docs)
        return self.tfidf_vectorizer.transform(clean_docs)

    def fit_transform_tfidf(self, documents):
        """Fit and transform documents into TF-IDF vectors."""
        if not documents:
            return None
        clean_docs = [doc if doc else "" for doc in documents]
        self._is_fitted = True
        return self.tfidf_vectorizer.fit_transform(clean_docs)

    def get_feature_names(self):
        """Get feature names from the fitted TF-IDF vectorizer."""
        if self._is_fitted:
            return self.tfidf_vectorizer.get_feature_names_out()
        return []

    def calculate_skill_frequency_matrix(self, candidates_skills, all_skills=None):
        """
        Create a binary skill frequency matrix.
        candidates_skills: list of skill lists per candidate
        all_skills: optional master skill list
        """
        if all_skills is None:
            all_skills = set()
            for skills in candidates_skills:
                all_skills.update(skills)
            all_skills = sorted(all_skills)

        matrix = np.zeros((len(candidates_skills), len(all_skills)))
        for i, skills in enumerate(candidates_skills):
            skill_set = set(s.lower().strip() for s in skills)
            for j, skill in enumerate(all_skills):
                if skill.lower() in skill_set:
                    matrix[i][j] = 1
        return matrix, list(all_skills)

    def calculate_experience_score(self, years):
        """Convert years of experience to a normalized score (0-1)."""
        if years <= 0:
            return 0.0
        # Logarithmic scaling: diminishing returns after certain years
        score = min(years / 15.0, 1.0)
        return round(score, 3)

    def calculate_education_score(self, education_text):
        """Convert education level to a normalized score (0-1)."""
        if not education_text:
            return 0.0
        edu_lower = education_text.lower()
        best_score = 0
        for level, score in EDUCATION_LEVELS.items():
            if level in edu_lower:
                best_score = max(best_score, score)
        return round(best_score / EDUCATION_MAX_SCORE, 3) if EDUCATION_MAX_SCORE > 0 else 0.0

    def calculate_skills_match_score(self, candidate_skills, required_skills):
        """Calculate the percentage of required skills matched by candidate."""
        if not required_skills:
            return 0.0
        candidate_set = set(s.lower().strip() for s in candidate_skills)
        required_set = set(s.lower().strip() for s in required_skills)
        if not required_set:
            return 0.0
        matched = candidate_set.intersection(required_set)
        return round(len(matched) / len(required_set), 3)

    def build_candidate_feature_vector(self, candidate_data):
        """
        Build a feature vector for a candidate for ML prediction.
        Returns: [experience_score, education_score, num_skills, skill_diversity]
        """
        exp_score = self.calculate_experience_score(candidate_data.get("experience", 0))
        edu_score = self.calculate_education_score(candidate_data.get("education", ""))

        skills = candidate_data.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        num_skills = min(len(skills) / 20.0, 1.0)  # Normalize by max 20 skills

        return [exp_score, edu_score, num_skills, len(skills)]

    def build_feature_matrix(self, candidates):
        """Build feature matrix for all candidates."""
        features = []
        for c in candidates:
            features.append(self.build_candidate_feature_vector(c))
        return np.array(features) if features else np.array([]).reshape(0, 4)

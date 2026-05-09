"""
SmartHire AI – Candidate Ranking
Multi-factor candidate ranking system combining similarity, skills, experience, and education.
"""

import pandas as pd
from modules.feature_engineering import FeatureEngineer
from utils.constants import RANKING_WEIGHTS


class CandidateRanker:
    """Ranks candidates using weighted multi-factor scoring."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.weights = RANKING_WEIGHTS

    def rank_candidates(self, candidates, job_required_skills=None):
        """
        Rank a list of candidate dicts by composite score.
        Each candidate dict should have: name, skills, education, experience,
        similarity_score.
        Returns sorted list with added 'rank' and 'composite_score' fields.
        """
        if not candidates:
            return []

        ranked = []
        for candidate in candidates:
            scores = self._calculate_component_scores(candidate, job_required_skills)
            composite = self._calculate_composite_score(scores)
            ranked_candidate = {**candidate, **scores, "composite_score": composite}
            ranked.append(ranked_candidate)

        # Sort by composite score descending
        ranked.sort(key=lambda x: x["composite_score"], reverse=True)

        # Add rank
        for i, candidate in enumerate(ranked):
            candidate["rank"] = i + 1

        return ranked

    def _calculate_component_scores(self, candidate, job_required_skills=None):
        """Calculate individual scoring components for a candidate."""
        # Similarity score (already computed)
        sim_score = candidate.get("similarity_score", 0.0)

        # Skills match score
        candidate_skills = candidate.get("skills", [])
        if isinstance(candidate_skills, str):
            candidate_skills = [s.strip() for s in candidate_skills.split(",") if s.strip()]
        if job_required_skills:
            req_skills = job_required_skills
            if isinstance(req_skills, str):
                req_skills = [s.strip() for s in req_skills.split(",") if s.strip()]
            skills_score = self.feature_engineer.calculate_skills_match_score(
                candidate_skills, req_skills
            )
        else:
            # Use skill count as a proxy
            skills_score = min(len(candidate_skills) / 10.0, 1.0)

        # Experience score
        exp_score = self.feature_engineer.calculate_experience_score(
            candidate.get("experience", 0)
        )

        # Education score
        edu_score = self.feature_engineer.calculate_education_score(
            candidate.get("education", "")
        )

        return {
            "sim_component": round(sim_score, 3),
            "skills_component": round(skills_score, 3),
            "experience_component": round(exp_score, 3),
            "education_component": round(edu_score, 3),
        }

    def _calculate_composite_score(self, scores):
        """Calculate weighted composite score."""
        composite = (
            scores.get("sim_component", 0) * self.weights["similarity_score"] +
            scores.get("skills_component", 0) * self.weights["skills_match"] +
            scores.get("experience_component", 0) * self.weights["experience_score"] +
            scores.get("education_component", 0) * self.weights["education_score"]
        )
        return round(composite, 4)

    def get_top_candidates(self, candidates, n=10, job_required_skills=None):
        """Get top N ranked candidates."""
        ranked = self.rank_candidates(candidates, job_required_skills)
        return ranked[:n]

    def get_ranking_dataframe(self, candidates, job_required_skills=None):
        """Return ranked candidates as a pandas DataFrame."""
        ranked = self.rank_candidates(candidates, job_required_skills)
        if not ranked:
            return pd.DataFrame()
        df = pd.DataFrame(ranked)
        display_cols = [
            "rank", "name", "email", "composite_score", "sim_component",
            "skills_component", "experience_component", "education_component",
            "experience", "education", "skills"
        ]
        available = [c for c in display_cols if c in df.columns]
        return df[available]

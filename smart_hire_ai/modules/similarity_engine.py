"""
SmartHire AI – Similarity Engine
Computes cosine similarity between resumes and job descriptions using TF-IDF.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from modules.feature_engineering import FeatureEngineer


class SimilarityEngine:
    """Computes similarity between candidate resumes and job descriptions."""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()

    def compute_similarity(self, resume_text, job_description):
        """
        Compute cosine similarity between a single resume and job description.
        Returns similarity score between 0 and 1.
        """
        if not resume_text or not job_description:
            return 0.0
        try:
            tfidf_matrix = self.feature_engineer.fit_transform_tfidf(
                [resume_text, job_description]
            )
            if tfidf_matrix is None:
                return 0.0
            sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(round(sim[0][0], 4))
        except Exception as e:
            print(f"Similarity computation error: {e}")
            return 0.0

    def compute_batch_similarity(self, resume_texts, job_description):
        """
        Compute similarity for multiple resumes against one job description.
        Returns list of similarity scores.
        """
        if not resume_texts or not job_description:
            return [0.0] * len(resume_texts) if resume_texts else []
        try:
            all_docs = resume_texts + [job_description]
            tfidf_matrix = self.feature_engineer.fit_transform_tfidf(all_docs)
            if tfidf_matrix is None:
                return [0.0] * len(resume_texts)
            # Job description is the last document
            job_vector = tfidf_matrix[-1:]
            resume_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(resume_vectors, job_vector)
            return [float(round(s[0], 4)) for s in similarities]
        except Exception as e:
            print(f"Batch similarity error: {e}")
            return [0.0] * len(resume_texts)

    def compute_similarity_matrix(self, resume_texts, candidate_names=None):
        """
        Compute pairwise similarity matrix between all resumes.
        Returns similarity matrix and names.
        """
        if not resume_texts:
            return np.array([]), []
        try:
            tfidf_matrix = self.feature_engineer.fit_transform_tfidf(resume_texts)
            if tfidf_matrix is None:
                return np.array([]), []
            sim_matrix = cosine_similarity(tfidf_matrix)
            names = candidate_names or [f"Candidate {i+1}" for i in range(len(resume_texts))]
            return np.round(sim_matrix, 3), names
        except Exception as e:
            print(f"Similarity matrix error: {e}")
            return np.array([]), []

    def get_top_matching_keywords(self, resume_text, job_description, top_n=10):
        """Get the top matching TF-IDF keywords between resume and job description."""
        if not resume_text or not job_description:
            return []
        try:
            fe = FeatureEngineer()
            tfidf_matrix = fe.fit_transform_tfidf([resume_text, job_description])
            if tfidf_matrix is None:
                return []
            feature_names = fe.get_feature_names()
            resume_scores = tfidf_matrix[0].toarray().flatten()
            job_scores = tfidf_matrix[1].toarray().flatten()
            # Find words important in both
            combined = resume_scores * job_scores
            top_indices = combined.argsort()[-top_n:][::-1]
            return [(feature_names[i], round(combined[i], 4))
                    for i in top_indices if combined[i] > 0]
        except Exception as e:
            print(f"Keyword matching error: {e}")
            return []

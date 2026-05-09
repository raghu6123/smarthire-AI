"""
SmartHire AI – Bias Detection
Basic fairness analysis for hiring bias detection.
"""

import numpy as np
from utils.constants import BIAS_THRESHOLDS


# Common first names for gender heuristic (simplified)
MALE_NAMES = {
    "james", "john", "robert", "michael", "david", "william", "richard",
    "joseph", "thomas", "charles", "daniel", "matthew", "anthony", "mark",
    "steven", "paul", "andrew", "joshua", "kenneth", "kevin", "brian",
    "george", "timothy", "ronald", "edward", "jason", "jeffrey", "ryan",
    "jacob", "gary", "nicholas", "eric", "jonathan", "stephen", "larry",
    "justin", "scott", "brandon", "benjamin", "samuel", "raymond", "greg",
    "frank", "alexander", "patrick", "jack", "dennis", "jerry", "tyler",
    "rahul", "amit", "raj", "vikram", "arjun", "sanjay", "rohit", "arun",
}

FEMALE_NAMES = {
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "margaret", "sandra", "ashley", "emily", "donna", "michelle",
    "dorothy", "carol", "amanda", "melissa", "deborah", "stephanie",
    "rebecca", "sharon", "laura", "cynthia", "kathleen", "amy", "angela",
    "shirley", "anna", "brenda", "pamela", "emma", "nicole", "helen",
    "katherine", "christine", "samantha", "rachel", "catherine", "hannah",
    "priya", "anjali", "neha", "pooja", "shreya", "divya", "swati", "anita",
}


class BiasDetector:
    """Detects potential hiring biases in candidate selection."""

    def __init__(self):
        self.thresholds = BIAS_THRESHOLDS

    def analyze_gender_distribution(self, candidates):
        """
        Analyze gender distribution based on name heuristic.
        Returns distribution dict and any warnings.
        """
        gender_counts = {"Male": 0, "Female": 0, "Unknown": 0}

        for c in candidates:
            gender = c.get("gender", "")
            if not gender:
                gender = self._infer_gender(c.get("name", ""))
            gender_counts[gender] = gender_counts.get(gender, 0) + 1

        total = sum(gender_counts.values())
        warnings = []

        if total > 0:
            for gender in ["Male", "Female"]:
                ratio = gender_counts.get(gender, 0) / total
                if ratio < self.thresholds["gender_skew_warning"] and gender_counts.get(gender, 0) > 0:
                    warnings.append(
                        f"Warning: {gender} candidates are underrepresented "
                        f"({ratio:.0%} of total). Consider broadening sourcing."
                    )

        return {
            "distribution": gender_counts,
            "total": total,
            "warnings": warnings,
        }

    def analyze_experience_bias(self, candidates, selected_ids=None):
        """
        Check if selection is biased toward certain experience levels.
        """
        exp_groups = {"Entry (0-2 yrs)": [], "Junior (2-5 yrs)": [],
                      "Mid (5-10 yrs)": [], "Senior (10+ yrs)": []}

        for c in candidates:
            exp = c.get("experience", 0) or 0
            cid = c.get("id", 0)
            if exp <= 2:
                exp_groups["Entry (0-2 yrs)"].append(cid)
            elif exp <= 5:
                exp_groups["Junior (2-5 yrs)"].append(cid)
            elif exp <= 10:
                exp_groups["Mid (5-10 yrs)"].append(cid)
            else:
                exp_groups["Senior (10+ yrs)"].append(cid)

        selected = set(selected_ids) if selected_ids else set()
        analysis = {}
        warnings = []
        total = len(candidates)

        for group, ids in exp_groups.items():
            count = len(ids)
            selected_count = len(set(ids) & selected) if selected else 0
            ratio = count / total if total > 0 else 0
            analysis[group] = {
                "total": count,
                "selected": selected_count,
                "ratio": round(ratio, 3),
            }
            if count > 0 and ratio < self.thresholds["experience_skew_warning"]:
                warnings.append(
                    f"Warning: '{group}' group is underrepresented ({ratio:.0%}). "
                    f"This could indicate experience bias in sourcing."
                )

        return {"analysis": analysis, "warnings": warnings}

    def analyze_skill_bias(self, candidates):
        """
        Detect if certain skill groups are over/under-represented.
        """
        from utils.constants import SKILL_CATEGORIES

        category_counts = {cat: 0 for cat in SKILL_CATEGORIES}
        total_skills = 0

        for c in candidates:
            skills = c.get("skills", "")
            if isinstance(skills, str):
                skills = [s.strip().lower() for s in skills.split(",") if s.strip()]
            else:
                skills = [s.lower() for s in skills]

            for cat, cat_skills in SKILL_CATEGORIES.items():
                matched = len(set(skills) & cat_skills)
                category_counts[cat] += matched
                total_skills += matched

        warnings = []
        analysis = {}

        for cat, count in category_counts.items():
            ratio = count / total_skills if total_skills > 0 else 0
            analysis[cat] = {"count": count, "ratio": round(ratio, 3)}
            if total_skills > 0 and ratio < self.thresholds["skill_bias_warning"] and count > 0:
                warnings.append(
                    f"Warning: '{cat}' skills are underrepresented ({ratio:.0%}). "
                    f"Consider diversifying skill requirements."
                )

        return {"analysis": analysis, "warnings": warnings}

    def get_fairness_summary(self, candidates, selected_ids=None):
        """Generate a complete fairness summary."""
        gender = self.analyze_gender_distribution(candidates)
        experience = self.analyze_experience_bias(candidates, selected_ids)
        skills = self.analyze_skill_bias(candidates)

        all_warnings = gender["warnings"] + experience["warnings"] + skills["warnings"]

        if not all_warnings:
            overall = "No significant bias detected. Distribution appears fair."
        elif len(all_warnings) <= 2:
            overall = "Minor bias indicators detected. Review recommended."
        else:
            overall = "Multiple bias indicators detected. Immediate review required."

        return {
            "gender": gender,
            "experience": experience,
            "skills": skills,
            "all_warnings": all_warnings,
            "overall_status": overall,
        }

    def _infer_gender(self, name):
        """Infer gender from first name (heuristic only)."""
        if not name:
            return "Unknown"
        first_name = name.split()[0].lower().strip() if name else ""
        if first_name in MALE_NAMES:
            return "Male"
        elif first_name in FEMALE_NAMES:
            return "Female"
        return "Unknown"

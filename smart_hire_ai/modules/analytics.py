"""
SmartHire AI – Analytics Dashboard
Interactive chart generation using Plotly for recruitment analytics.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils.constants import PLOTLY_COLORS, PLOTLY_TEMPLATE


class AnalyticsDashboard:
    """Generates interactive Plotly charts for recruitment analytics."""

    def __init__(self):
        self.colors = PLOTLY_COLORS
        self.template = PLOTLY_TEMPLATE

    def _apply_dark_layout(self, fig, title=""):
        """Apply consistent dark theme to all charts."""
        fig.update_layout(
            title=dict(text=title, font=dict(size=18, color="#f1f5f9")),
            template=self.template,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=12),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def candidate_ranking_chart(self, ranked_candidates, top_n=10):
        """Horizontal bar chart of top ranked candidates."""
        if not ranked_candidates:
            return self._empty_chart("No ranking data available")
        df = pd.DataFrame(ranked_candidates[:top_n]).sort_values("composite_score", ascending=True)
        fig = go.Figure(go.Bar(
            x=df["composite_score"], y=df["name"], orientation='h',
            marker=dict(color=df["composite_score"],
                        colorscale=[[0, "#6366f1"], [0.5, "#8b5cf6"], [1, "#06b6d4"]]),
            text=[f"{s:.0%}" for s in df["composite_score"]],
            textposition="auto", textfont=dict(color="white"),
        ))
        return self._apply_dark_layout(fig, "Top Candidate Rankings")

    def skill_frequency_chart(self, skill_distribution, top_n=15):
        """Bar chart of most common skills."""
        if not skill_distribution:
            return self._empty_chart("No skill data available")
        items = sorted(skill_distribution.items(), key=lambda x: x[1], reverse=True)[:top_n]
        skills, counts = zip(*items)
        fig = go.Figure(go.Bar(
            x=list(counts), y=list(skills), orientation='h',
            marker=dict(color=list(range(len(skills))), colorscale="Viridis"),
            text=list(counts), textposition="auto", textfont=dict(color="white"),
        ))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return self._apply_dark_layout(fig, "Top Skills Distribution")

    def experience_distribution_chart(self, exp_distribution):
        """Donut chart of experience levels."""
        if not exp_distribution:
            return self._empty_chart("No experience data")
        fig = go.Figure(go.Pie(
            labels=list(exp_distribution.keys()), values=list(exp_distribution.values()),
            hole=0.4, marker=dict(colors=self.colors),
            textinfo="label+percent", textfont=dict(color="white", size=13),
        ))
        return self._apply_dark_layout(fig, "Experience Distribution")

    def department_analysis_chart(self, dept_distribution):
        """Bar chart of department-wise candidates."""
        if not dept_distribution:
            return self._empty_chart("No department data")
        fig = go.Figure(go.Bar(
            x=list(dept_distribution.keys()), y=list(dept_distribution.values()),
            marker=dict(color=self.colors[:len(dept_distribution)]),
            text=list(dept_distribution.values()), textposition="auto",
            textfont=dict(color="white"),
        ))
        return self._apply_dark_layout(fig, "Department-wise Distribution")

    def hiring_probability_chart(self, candidates):
        """Scatter plot of hiring probability vs experience."""
        if not candidates:
            return self._empty_chart("No candidate data")
        df = pd.DataFrame(candidates)
        if "suitability_score" not in df.columns:
            return self._empty_chart("Missing suitability data")
        fig = px.scatter(
            df, x="experience", y="suitability_score", size="suitability_score",
            color="suitability_score", hover_name="name" if "name" in df.columns else None,
            color_continuous_scale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
            size_max=20,
        )
        return self._apply_dark_layout(fig, "Hiring Probability vs Experience")

    def diversity_analysis_chart(self, gender_distribution):
        """Donut chart for gender diversity."""
        if not gender_distribution:
            return self._empty_chart("No diversity data")
        colors_map = {"Male": "#3b82f6", "Female": "#ec4899", "Unknown": "#64748b"}
        labels = list(gender_distribution.keys())
        fig = go.Figure(go.Pie(
            labels=labels, values=list(gender_distribution.values()),
            hole=0.5, marker=dict(colors=[colors_map.get(l, "#64748b") for l in labels]),
            textinfo="label+percent+value", textfont=dict(color="white", size=13),
        ))
        return self._apply_dark_layout(fig, "Gender Diversity Analysis")

    def similarity_heatmap(self, sim_matrix, names):
        """Heatmap of candidate similarity matrix."""
        if sim_matrix is None or len(sim_matrix) == 0:
            return self._empty_chart("No similarity data")
        fig = go.Figure(go.Heatmap(
            z=sim_matrix, x=names, y=names,
            colorscale=[[0, "#0a0e1a"], [0.5, "#6366f1"], [1, "#06b6d4"]],
            text=np.round(sim_matrix, 2), texttemplate="%{text}",
            textfont=dict(size=10, color="white"),
        ))
        return self._apply_dark_layout(fig, "Candidate Similarity Heatmap")

    def candidate_comparison_chart(self, candidates, metrics=None):
        """Radar chart comparing top candidates."""
        if not candidates or len(candidates) < 2:
            return self._empty_chart("Need at least 2 candidates")
        if metrics is None:
            metrics = ["sim_component", "skills_component", "experience_component", "education_component"]
        labels = ["Similarity", "Skills", "Experience", "Education"]
        fig = go.Figure()
        for i, c in enumerate(candidates[:5]):
            vals = [c.get(m, 0) for m in metrics] + [c.get(metrics[0], 0)]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels + [labels[0]], fill='toself',
                name=c.get("name", f"Candidate {i+1}"),
                line=dict(color=self.colors[i % len(self.colors)]),
            ))
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(99,102,241,0.2)"),
            angularaxis=dict(gridcolor="rgba(99,102,241,0.2)"),
        ))
        return self._apply_dark_layout(fig, "Candidate Comparison")

    def roc_curve_chart(self, roc_data):
        """ROC curve chart for ML models."""
        if not roc_data:
            return self._empty_chart("No ROC data available")
        fig = go.Figure()
        for i, (name, data) in enumerate(roc_data.items()):
            fig.add_trace(go.Scatter(
                x=data["fpr"], y=data["tpr"], mode='lines',
                name=f'{name} (AUC={data["auc"]:.3f})',
                line=dict(color=self.colors[i % len(self.colors)], width=2),
            ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color='gray', width=1, dash='dash'),
        ))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        return self._apply_dark_layout(fig, "ROC Curve Comparison")

    def confusion_matrix_chart(self, cm, model_name=""):
        """Confusion matrix heatmap."""
        if cm is None:
            return self._empty_chart("No confusion matrix data")
        labels = ["Not Suitable", "Suitable"]
        cm_arr = np.array(cm)
        fig = go.Figure(go.Heatmap(
            z=cm_arr, x=labels, y=labels,
            colorscale=[[0, "#1e1b4b"], [1, "#6366f1"]],
            text=cm_arr, texttemplate="%{text}", textfont=dict(size=18, color="white"),
        ))
        fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
        return self._apply_dark_layout(fig, f"Confusion Matrix – {model_name}")

    def skill_gap_chart(self, candidate_skills, required_skills):
        """Skill gap analysis chart."""
        if not required_skills:
            return self._empty_chart("No required skills provided")
        req = set(s.lower().strip() for s in required_skills)
        has = set(s.lower().strip() for s in candidate_skills)
        matched = req & has
        missing = req - has
        categories, values, colors = [], [], []
        for s in sorted(matched):
            categories.append(s.title()); values.append(1); colors.append("#10b981")
        for s in sorted(missing):
            categories.append(s.title()); values.append(-1); colors.append("#ef4444")
        fig = go.Figure(go.Bar(
            x=values, y=categories, orientation='h', marker=dict(color=colors),
        ))
        fig.update_layout(xaxis=dict(tickvals=[-1, 1], ticktext=["Missing", "Matched"]))
        return self._apply_dark_layout(fig, "Skill Gap Analysis")

    def _empty_chart(self, message):
        """Empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(text=message, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color="#64748b"))
        return self._apply_dark_layout(fig)

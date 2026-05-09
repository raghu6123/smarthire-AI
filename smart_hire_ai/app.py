"""
SmartHire AI – Main Application
Streamlit frontend with premium UI for the recruitment system.
"""

import streamlit as st
import pandas as pd
import os
import time

# --- Module Imports ---
from utils.constants import PAGES, COLORS, SAMPLE_DEPARTMENTS, SAMPLE_JOB_ROLES
from utils.helpers import (
    create_skill_tags_html, get_score_color, get_score_label,
    dataframe_to_csv_bytes, truncate_text
)
from modules.database_manager import DatabaseManager
from modules.parser import ResumeParser
from modules.similarity_engine import SimilarityEngine
from modules.ranking import CandidateRanker
from modules.classifier import SuitabilityClassifier
from modules.bias_detection import BiasDetector
from modules.analytics import AnalyticsDashboard
from modules.report_generator import ReportGenerator
from modules.feature_engineering import FeatureEngineer

# --- Configuration & Styling ---
st.set_page_config(
    page_title="SmartHire AI",
    page_icon="SmartHire",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown(f"""
    <style>
        /* General dark theme base */
        .stApp {{
            background-color: {COLORS['bg_primary']};
            color: {COLORS['text_primary']};
            font-family: 'Inter', 'Roboto', sans-serif;
        }}
        
        /* Glassmorphism Cards */
        .glass-card {{
            background: {COLORS['bg_card']};
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid {COLORS['border']};
            padding: 24px;
            box-shadow: 0 4px 6px {COLORS['shadow']};
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px rgba(0,0,0,0.4);
        }}
        
        /* Typography */
        h1, h2, h3, h4 {{
            color: {COLORS['text_primary']} !important;
            font-weight: 600;
        }}
        p, span, div {{
            color: {COLORS['text_secondary']};
        }}
        
        /* Gradient Accents */
        .gradient-text {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }}
        
        /* Metric display */
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin: 0;
            line-height: 1.2;
        }}
        .metric-label {{
            font-size: 1rem;
            color: {COLORS['text_muted']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1dp5vir {{
            background-color: {COLORS['bg_secondary']} !important;
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        
        /* Dataframes */
        .stDataFrame {{
            background: transparent !important;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 24px;
            background-color: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: transparent !important;
            border-bottom: 2px solid {COLORS['accent_primary']} !important;
            color: {COLORS['accent_primary']} !important;
        }}
        
        /* Buttons */
        .stButton>button {{
            background: linear-gradient(135deg, {COLORS['gradient_start']}, {COLORS['gradient_end']});
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: opacity 0.2s ease;
        }}
        .stButton>button:hover {{
            opacity: 0.9;
            color: white;
        }}
    </style>
    """, unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource
def get_db_manager():
    return DatabaseManager()

@st.cache_resource
def get_classifier():
    classifier = SuitabilityClassifier()
    if not classifier.is_trained:
        # Load or train dummy on startup if needed, though usually user triggers training
        try:
            classifier._load_model()
        except:
            pass
    return classifier

db = get_db_manager()
classifier = get_classifier()
analytics = AnalyticsDashboard()

# Create sample jobs if DB is empty
if len(db.get_all_jobs()) == 0:
    for i in range(3):
         db.add_job(
             role=SAMPLE_JOB_ROLES[i],
             required_skills="Python, SQL, Machine Learning, Data Analysis",
             description="Looking for an experienced professional to join our team.",
             department=SAMPLE_DEPARTMENTS[i]
         )

# --- Helper Components ---
def render_metric_card(title, value, subtitle="", col=None):
    html = f"""
    <div class="glass-card" style="text-align: center; padding: 1.5rem;">
        <div class="metric-label">{title}</div>
        <div class="metric-value gradient-text">{value}</div>
        <div style="font-size: 0.85rem; color: {COLORS['success']}; margin-top: 0.5rem;">{subtitle}</div>
    </div>
    """
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)

# --- Page Renderers ---

def render_dashboard():
    st.markdown('<h1 class="gradient-text">SmartHire AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<p>Welcome to the Intelligent Recruitment & Candidate Analytics System.</p>", unsafe_allow_html=True)
    
    candidates = db.get_all_candidates()
    jobs = db.get_all_jobs()
    
    col1, col2, col3, col4 = st.columns(4)
    render_metric_card("Total Candidates", len(candidates), "↑ 12% this week", col1)
    render_metric_card("Active Jobs", len(jobs), "Currently hiring", col2)
    
    avg_sim = db.get_avg_similarity_score()
    render_metric_card("Avg Match Score", f"{avg_sim:.0%}", "Overall quality", col3)
    
    top_skills = db.get_skill_distribution()
    top_skill_name = list(top_skills.keys())[0].title() if top_skills else "N/A"
    render_metric_card("Top Skill Demand", top_skill_name, "Most common", col4)
    
    st.markdown("---")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### Recent Activity")
        if candidates:
            df = pd.DataFrame(candidates[:5])[["name", "department", "experience", "created_at"]]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No candidates uploaded yet. Go to 'Upload Resumes' to get started.")
            
    with col_b:
        st.markdown("### Quick Actions")
        if st.button("Upload New Resumes", use_container_width=True):
            st.session_state.page = "Upload Resumes"
            st.rerun()
        if st.button("Run Job Match", use_container_width=True):
            st.session_state.page = "Job Matching"
            st.rerun()
        if st.button("Generate Report", use_container_width=True):
            st.session_state.page = "Report Generator"
            st.rerun()

def render_upload_resumes():
    st.markdown('<h1 class="gradient-text">Upload Resumes</h1>', unsafe_allow_html=True)
    st.markdown("Upload candidate resumes (PDF/DOCX) for automatic parsing and extraction.")
    
    parser = ResumeParser()
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Select Resumes", type=['pdf', 'docx'], accept_multiple_files=True)
        
        selected_dept = st.selectbox("Assign to Department (Optional)", [""] + SAMPLE_DEPARTMENTS)
        
        if st.button("Process Resumes") and uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            parsed_count = 0
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Parsing {file.name} ({i+1}/{len(uploaded_files)})...")
                try:
                    info = parser.parse_uploaded_file(file)
                    
                    # Basic gender inference for demo
                    from modules.bias_detection import BiasDetector
                    bd = BiasDetector()
                    gender = bd._infer_gender(info['name'])
                    
                    db.add_candidate(
                        name=info['name'],
                        email=info['email'],
                        skills=",".join(info['skills']),
                        education=info['education'],
                        experience=info['experience'],
                        resume_text=info['raw_text'],
                        resume_path=file.name,
                        department=selected_dept,
                        gender=gender
                    )
                    parsed_count += 1
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            st.success(f"Successfully processed and stored {parsed_count} resumes.")
            time.sleep(1)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def render_job_matching():
    st.markdown('<h1 class="gradient-text">Job Description Matching</h1>', unsafe_allow_html=True)
    
    candidates = db.get_all_candidates()
    if not candidates:
        st.warning("Please upload some resumes first.")
        return
        
    jobs = db.get_all_jobs()
    job_options = {f"{j['role']} ({j['department']})": j for j in jobs}
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Select Job")
        selected_job_name = st.selectbox("Job Profile", list(job_options.keys()))
        job = job_options[selected_job_name]
        
        st.markdown(f"**Required Skills:** {job['required_skills']}")
        with st.expander("View Job Description"):
            st.write(job['description'])
            
        if st.button("Run Similarity Analysis"):
            with st.spinner("Computing TF-IDF & Cosine Similarities..."):
                engine = SimilarityEngine()
                resume_texts = [c['resume_text'] for c in candidates]
                desc = job['description'] + " " + job['required_skills']
                
                similarities = engine.compute_batch_similarity(resume_texts, desc)
                
                for cid, sim in zip([c['id'] for c in candidates], similarities):
                    db.update_candidate_scores(cid, similarity_score=sim)
                
                st.success("Analysis complete!")
                st.rerun()

    with col2:
        st.markdown("### Match Results")
        # Fetch updated candidates
        cands = db.get_all_candidates()
        # Filter those with scores > 0
        matched = [c for c in cands if c['similarity_score'] > 0]
        matched.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        if matched:
            for c in matched[:5]:  # Show top 5
                score = c['similarity_score']
                color = get_score_color(score)
                label = get_score_label(score)
                
                st.markdown(f"""
                <div class="glass-card" style="margin-bottom: 1rem; padding: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h4 style="margin:0;">{c['name']}</h4>
                            <span style="font-size: 0.9rem; color: #94a3b8;">{c['experience']} yrs exp • {c['education']}</span>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">{score:.0%}</div>
                            <div style="font-size: 0.8rem; color: {color};">{label} Match</div>
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem;">
                        {create_skill_tags_html(c['skills'], 5)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run analysis to see match results.")

def render_candidate_ranking():
    st.markdown('<h1 class="gradient-text">Candidate Ranking System</h1>', unsafe_allow_html=True)
    
    candidates = db.get_all_candidates()
    if not candidates:
        st.warning("No candidates available.")
        return
        
    ranker = CandidateRanker()
    ranked = ranker.rank_candidates(candidates)
    
    # Display Leaderboard
    st.markdown("### Top Candidates Leaderboard")
    df = pd.DataFrame(ranked)
    display_df = df[['rank', 'name', 'composite_score', 'similarity_score', 'experience', 'education']]
    
    # Format scores for display
    display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.2f}")
    display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.0%}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.download_button(
        label="Download Ranking CSV",
        data=dataframe_to_csv_bytes(df),
        file_name="candidate_rankings.csv",
        mime="text/csv"
    )

def render_suitability_prediction():
    st.markdown('<h1 class="gradient-text">ML Suitability Prediction</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### Model Training")
        st.write("Train ML models to predict candidate suitability based on historical hiring patterns (using generated synthetic data for demo).")
        
        if st.button("Train / Retrain Models"):
            with st.spinner("Training Logistic Regression, Random Forest, and Gradient Boosting..."):
                metrics = classifier.train_models()
                st.success(f"Best Model: {classifier.best_model_name}")
                st.session_state.model_metrics = metrics
        st.markdown('</div>', unsafe_allow_html=True)
        
        if classifier.is_trained:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Predict for Current Candidates")
            if st.button("Run Predictions"):
                cands = db.get_all_candidates()
                if cands:
                    fe = FeatureEngineer()
                    features = fe.build_feature_matrix(cands)
                    preds = classifier.predict_batch(features)
                    
                    for c, p in zip(cands, preds):
                        db.update_candidate_scores(c['id'], suitability_score=p['probability'])
                        db.add_prediction(c['id'], p['prediction'], p['probability'], classifier.best_model_name)
                    st.success("Predictions completed and saved!")
                else:
                    st.warning("No candidates to predict.")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if classifier.is_trained:
            st.markdown("### Model Performance Metrics")
            metrics_df = classifier.get_metrics_dataframe()
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            if hasattr(classifier, 'roc_data') and classifier.roc_data:
                st.plotly_chart(analytics.roc_curve_chart(classifier.roc_data), use_container_width=True)
        else:
            st.info("Train models to view performance metrics.")

def render_bias_detection():
    st.markdown('<h1 class="gradient-text">Bias & Fairness Detection</h1>', unsafe_allow_html=True)
    
    candidates = db.get_all_candidates()
    if not candidates:
        st.warning("No candidates available for analysis.")
        return
        
    detector = BiasDetector()
    summary = detector.get_fairness_summary(candidates)
    
    # Overall Status Banner
    status_color = "#10b981" if "No significant" in summary['overall_status'] else "#f59e0b" if "Minor bias" in summary['overall_status'] else "#ef4444"
    st.markdown(f"""
    <div style="background-color: rgba({int(status_color[1:3],16)}, {int(status_color[3:5],16)}, {int(status_color[5:7],16)}, 0.1); 
                border-left: 4px solid {status_color}; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;">
        <h3 style="margin: 0; color: {status_color};">{summary['overall_status']}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(analytics.diversity_analysis_chart(summary['gender']['distribution']), use_container_width=True)
    with col2:
        st.markdown("### Warnings")
        if summary['all_warnings']:
            for w in summary['all_warnings']:
                st.error(w)
        else:
            st.success("No warnings detected.")

def render_analytics():
    st.markdown('<h1 class="gradient-text">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    candidates = db.get_all_candidates()
    if not candidates:
        st.warning("Insufficient data for analytics.")
        return
        
    ranker = CandidateRanker()
    ranked = ranker.rank_candidates(candidates)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(analytics.candidate_ranking_chart(ranked), use_container_width=True)
        st.plotly_chart(analytics.experience_distribution_chart(db.get_experience_distribution()), use_container_width=True)
    with col2:
        st.plotly_chart(analytics.skill_frequency_chart(db.get_skill_distribution()), use_container_width=True)
        st.plotly_chart(analytics.department_analysis_chart(db.get_department_distribution()), use_container_width=True)
        
    st.plotly_chart(analytics.hiring_probability_chart(candidates), use_container_width=True)

def render_reports():
    st.markdown('<h1 class="gradient-text">Report Generator</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("Generate a comprehensive PDF report containing rankings, analytics, and bias detection results.")
    
    report_name = st.text_input("Report Filename", value=f"Recruitment_Report_{time.strftime('%Y%m%d')}.pdf")
    
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            candidates = db.get_all_candidates()
            ranker = CandidateRanker()
            ranked = ranker.rank_candidates(candidates)
            
            bd = BiasDetector()
            bias_summary = bd.get_fairness_summary(candidates)
            
            rg = ReportGenerator()
            filepath = rg.generate_recruitment_report(
                candidates=ranked,
                bias_summary=bias_summary,
                ml_metrics=classifier.metrics if classifier.is_trained else None,
                filename=report_name
            )
            
            st.success(f"Report generated successfully!")
            
            with open(filepath, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF",
                    data=pdf_file,
                    file_name=report_name,
                    mime="application/pdf"
                )
    st.markdown('</div>', unsafe_allow_html=True)

# --- Navigation & Main ---
def main():
    inject_custom_css()
    
    # Sidebar Navigation
    st.sidebar.markdown('<h2 class="gradient-text">SmartHire AI</h2>', unsafe_allow_html=True)
    st.sidebar.markdown("Intelligent Recruitment System")
    st.sidebar.markdown("---")
    
    # Init session state for navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Dashboard"
        
    for name, internal_name in PAGES.items():
        if st.sidebar.button(name, use_container_width=True, 
                             type="primary" if st.session_state.page == name else "secondary"):
            st.session_state.page = name
            st.rerun()
            
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Database"):
        db.clear_all_data()
        st.sidebar.success("Database cleared!")
        time.sleep(1)
        st.rerun()

    # Route to page
    page = st.session_state.page
    if page == "Dashboard": render_dashboard()
    elif page == "Upload Resumes": render_upload_resumes()
    elif page == "Job Matching": render_job_matching()
    elif page == "Candidate Ranking": render_candidate_ranking()
    elif page == "Suitability Prediction": render_suitability_prediction()
    elif page == "Bias Detection": render_bias_detection()
    elif page == "Analytics Dashboard": render_analytics()
    elif page == "Report Generator": render_reports()

if __name__ == "__main__":
    main()

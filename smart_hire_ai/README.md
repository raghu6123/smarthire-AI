# SmartHire AI – Intelligent Recruitment & Candidate Analytics System

SmartHire AI is an AI-powered applicant tracking and recruitment analytics platform. It automates resume parsing, scores candidate suitability using Machine Learning and NLP, and provides a premium analytics dashboard to detect hiring biases and generate recruitment reports.

## Features

- **Resume Parsing:** Upload PDF or DOCX resumes to automatically extract names, contact info, skills, education, and experience.
- **Job Matching Engine:** Uses TF-IDF and Cosine Similarity to score candidate resumes against job descriptions.
- **Candidate Ranking:** A multi-factor ranking system considering similarity scores, skill matches, education, and experience.
- **Suitability Prediction (ML):** Trains Logistic Regression, Random Forest, and Gradient Boosting models on historical hiring data to predict candidate success probabilities.
- **Bias & Fairness Detection:** Analyzes candidate pools for potential gender, experience, and skill-based biases.
- **Interactive Analytics:** A rich dashboard built with Plotly showcasing skill frequencies, diversity, and similarity heatmaps.
- **Report Generation:** Export comprehensive PDF recruitment reports summarizing rankings, biases, and model metrics.

## Tech Stack

- **Frontend:** Streamlit
- **Backend/Logic:** Python
- **Database:** SQLite
- **NLP & ML:** Scikit-learn, NLTK, PyPDF2, python-docx
- **Visualizations:** Plotly
- **Reporting:** ReportLab

## Project Structure

```
smart_hire_ai/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── database/                  # SQLite database storage
├── data/                      # Resumes and processed data
├── models/                    # Serialized ML models (.pkl)
├── reports/                   # Generated PDF reports
├── assets/                    # Static assets
├── modules/                   # Core business logic
│   ├── parser.py              # PDF/DOCX extraction
│   ├── preprocessing.py       # NLP text cleaning
│   ├── feature_engineering.py # TF-IDF and scoring
│   ├── similarity_engine.py   # Cosine similarity logic
│   ├── ranking.py             # Composite scoring
│   ├── classifier.py          # ML suitability prediction
│   ├── bias_detection.py      # Fairness analysis
│   ├── analytics.py           # Plotly chart generation
│   ├── database_manager.py    # SQLite CRUD operations
│   └── report_generator.py    # PDF report creation
└── utils/
    ├── constants.py           # Configuration and thresholds
    └── helpers.py             # Reusable utility functions
```

## Setup & Installation

1. Clone or download the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open the provided local URL in your browser to access the SmartHire AI dashboard.

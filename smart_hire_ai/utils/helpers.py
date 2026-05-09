"""
SmartHire AI – Helper Utilities
Reusable utility functions for file handling, text processing, and formatting.
"""

import os
import re
import hashlib
from datetime import datetime


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_file(uploaded_file, destination_dir):
    """Save a Streamlit uploaded file to the specified directory."""
    ensure_directory(destination_dir)
    file_path = os.path.join(destination_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_file_extension(filename):
    """Get lowercase file extension."""
    return os.path.splitext(filename)[1].lower()


def clean_text(text):
    """Basic text cleaning."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;:!?@\-/()]+', '', text)
    return text.strip()


def extract_email(text):
    """Extract email address from text."""
    if not text:
        return ""
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(pattern, text)
    return match.group(0) if match else ""


def extract_phone(text):
    """Extract phone number from text."""
    if not text:
        return ""
    pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}'
    match = re.search(pattern, text)
    return match.group(0).strip() if match else ""


def extract_years_of_experience(text):
    """Extract years of experience from resume text."""
    if not text:
        return 0
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
        r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:in|of|working)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return min(int(match.group(1)), 50)
    return 0


def normalize_score(score, min_val=0, max_val=1):
    """Normalize a score to a 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0, min((score - min_val) / (max_val - min_val), 1))


def format_percentage(value, decimals=1):
    """Format a decimal value as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_score(value, decimals=2):
    """Format a score value."""
    return f"{value:.{decimals}f}"


def get_score_color(score):
    """Return a color based on score value (0-1 range)."""
    if score >= 0.8:
        return "#10b981"
    elif score >= 0.6:
        return "#3b82f6"
    elif score >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def get_score_label(score):
    """Return a label based on score value (0-1 range)."""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    return "Poor"


def dataframe_to_csv_bytes(df):
    """Convert a pandas DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


def format_datetime(dt=None):
    """Format datetime for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%B %d, %Y at %I:%M %p")


def truncate_text(text, max_length=100):
    """Truncate text with ellipsis."""
    if not text or len(text) <= max_length:
        return text or ""
    return text[:max_length - 3] + "..."


def extract_name_from_text(text):
    """Extract a name from the first few lines of resume text."""
    if not text:
        return "Unknown"
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if not line:
            continue
        if any(s in line.lower() for s in ['resume', 'curriculum', 'cv', 'http', 'www', '@']):
            continue
        if re.search(r'[\d]{5,}', line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(re.match(r'^[a-zA-Z.\'-]+$', w) for w in words):
            return line.title()
    return "Unknown"


def safe_divide(numerator, denominator, default=0):
    """Safely divide two numbers."""
    return numerator / denominator if denominator != 0 else default


def create_skill_tags_html(skills, max_display=10):
    """Create HTML for skill tags display."""
    if not skills:
        return "<span style='color: #64748b;'>No skills detected</span>"
    skill_list = [s.strip() for s in skills.split(',') if s.strip()] if isinstance(skills, str) else skills
    tags = []
    for skill in skill_list[:max_display]:
        tags.append(
            f'<span style="background: linear-gradient(135deg, #6366f1, #8b5cf6); '
            f'color: white; padding: 4px 12px; border-radius: 20px; '
            f'font-size: 0.8rem; margin: 2px; display: inline-block;">{skill}</span>'
        )
    if len(skill_list) > max_display:
        tags.append(f'<span style="color: #94a3b8; font-size: 0.8rem;">+{len(skill_list) - max_display} more</span>')
    return ' '.join(tags)

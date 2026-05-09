"""
SmartHire AI – Text Preprocessing
NLP pipeline for cleaning, tokenizing, and processing resume/job text.
"""

import re
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    # Download required NLTK data silently
    for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from utils.constants import SKILL_DICTIONARY


class TextPreprocessor:
    """NLP text preprocessing pipeline for resumes and job descriptions."""

    def __init__(self):
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words('english'))
            except Exception:
                self.stop_words = set()
        else:
            self.lemmatizer = None
            self.stop_words = set()

    def preprocess(self, text):
        """Full preprocessing pipeline: clean → tokenize → remove stops → lemmatize."""
        if not text:
            return ""
        text = self._lowercase(text)
        text = self._remove_urls(text)
        text = self._remove_special_chars(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)
        tokens = self._lemmatize(tokens)
        return ' '.join(tokens)

    def extract_keywords(self, text, top_n=20):
        """Extract top keywords from text using frequency analysis."""
        processed = self.preprocess(text)
        tokens = processed.split()
        # Filter short tokens and numbers
        tokens = [t for t in tokens if len(t) > 2 and not t.isdigit()]
        # Count frequency
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_keywords[:top_n]]

    def extract_skills_from_text(self, text):
        """Extract skills by matching against the custom skill dictionary."""
        if not text:
            return []
        text_lower = text.lower()
        found_skills = []
        for skill in SKILL_DICTIONARY:
            if len(skill) <= 3:
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.append(skill)
            else:
                if skill in text_lower:
                    found_skills.append(skill)
        return list(set(found_skills))

    def _lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def _remove_urls(self, text):
        """Remove URLs from text."""
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def _remove_special_chars(self, text):
        """Remove special characters, keeping spaces and basic punctuation."""
        return re.sub(r'[^\w\s]', ' ', text)

    def _tokenize(self, text):
        """Tokenize text into words."""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                return text.split()
        return text.split()

    def _remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        if not self.stop_words:
            return tokens
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def _lemmatize(self, tokens):
        """Lemmatize tokens."""
        if self.lemmatizer is None:
            return tokens
        try:
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            return tokens

    def get_text_stats(self, text):
        """Get basic text statistics."""
        if not text:
            return {"word_count": 0, "char_count": 0, "sentence_count": 0}
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        return {
            "word_count": len(words),
            "char_count": len(text),
            "sentence_count": len([s for s in sentences if s.strip()]),
        }

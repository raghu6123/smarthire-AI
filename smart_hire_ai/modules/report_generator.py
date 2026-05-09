"""
SmartHire AI – Report Generator
Generates PDF recruitment reports using ReportLab.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from utils.constants import REPORTS_DIR


class ReportGenerator:
    """Generates PDF reports for recruitment analytics."""

    def __init__(self):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1e1b4b'),
            alignment=1  # Center
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#4f46e5')
        ))
        self.styles.add(ParagraphStyle(
            name='NormalText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            textColor=colors.HexColor('#334155')
        ))
        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#ef4444'),
            spaceAfter=8
        ))

    def generate_recruitment_report(self, candidates, job_info=None, bias_summary=None, ml_metrics=None, filename=None):
        """
        Generate a comprehensive recruitment report.
        Returns the path to the generated PDF.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Recruitment_Report_{timestamp}.pdf"
        
        filepath = os.path.join(REPORTS_DIR, filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        elements = []
        
        # 1. Header & Title
        elements.append(Paragraph("SmartHire AI", self.styles['ReportTitle']))
        elements.append(Paragraph(f"Recruitment Analytics Report", self.styles['Heading2']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", self.styles['NormalText']))
        elements.append(Spacer(1, 20))
        
        # 2. Executive Summary
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        total_candidates = len(candidates)
        elements.append(Paragraph(f"Total Candidates Evaluated: {total_candidates}", self.styles['NormalText']))
        if job_info:
            elements.append(Paragraph(f"Target Role: {job_info.get('role', 'N/A')}", self.styles['NormalText']))
            elements.append(Paragraph(f"Department: {job_info.get('department', 'N/A')}", self.styles['NormalText']))
        elements.append(Spacer(1, 10))

        # 3. Top Candidates
        elements.append(Paragraph("Top Ranked Candidates", self.styles['SectionHeader']))
        if candidates:
            # Sort if not already sorted
            sorted_cands = sorted(candidates, key=lambda x: x.get('composite_score', x.get('similarity_score', 0)), reverse=True)
            top_n = min(10, len(sorted_cands))
            
            table_data = [['Rank', 'Name', 'Score', 'Exp (Yrs)', 'Match']]
            for i, c in enumerate(sorted_cands[:top_n]):
                score = c.get('composite_score', c.get('similarity_score', 0))
                table_data.append([
                    str(i + 1),
                    c.get('name', 'Unknown'),
                    f"{score:.2f}",
                    str(c.get('experience', 0)),
                    "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
                ])
                
            t = Table(table_data, colWidths=[40, 150, 60, 80, 80])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1'))
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("No candidates available.", self.styles['NormalText']))
        elements.append(Spacer(1, 20))

        # 4. Bias Detection Summary
        elements.append(Paragraph("Fairness & Bias Analysis", self.styles['SectionHeader']))
        if bias_summary:
            elements.append(Paragraph(f"Overall Status: {bias_summary.get('overall_status', 'N/A')}", self.styles['NormalText']))
            warnings = bias_summary.get('all_warnings', [])
            if warnings:
                elements.append(Paragraph("Detected Warnings:", self.styles['WarningText']))
                for w in warnings:
                    elements.append(Paragraph(f"• {w}", self.styles['WarningText']))
            else:
                elements.append(Paragraph("No significant bias detected.", self.styles['NormalText']))
        else:
            elements.append(Paragraph("Bias analysis not performed.", self.styles['NormalText']))
        elements.append(Spacer(1, 20))

        # 5. ML Model Performance (if available)
        if ml_metrics:
            elements.append(Paragraph("ML Model Performance", self.styles['SectionHeader']))
            model_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1']]
            for model_name, metrics in ml_metrics.items():
                model_data.append([
                    model_name,
                    f"{metrics.get('accuracy', 0):.2f}",
                    f"{metrics.get('precision', 0):.2f}",
                    f"{metrics.get('recall', 0):.2f}",
                    f"{metrics.get('f1_score', 0):.2f}"
                ])
            
            t2 = Table(model_data)
            t2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1'))
            ]))
            elements.append(t2)
            elements.append(Spacer(1, 20))

        # 6. Conclusion
        elements.append(Paragraph("Hiring Recommendation", self.styles['SectionHeader']))
        if candidates and len(candidates) > 0:
            top_cand = sorted_cands[0]
            elements.append(Paragraph(
                f"Based on the composite ranking algorithm, <b>{top_cand.get('name', 'Unknown')}</b> "
                f"is the highest recommended candidate with a score of {top_cand.get('composite_score', top_cand.get('similarity_score', 0)):.2f}.", 
                self.styles['NormalText']
            ))
        else:
            elements.append(Paragraph("Insufficient data to make a recommendation.", self.styles['NormalText']))

        # Build PDF
        doc.build(elements)
        return filepath

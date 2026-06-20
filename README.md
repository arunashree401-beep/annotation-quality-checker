# Annotation Quality Checker
**By Arunashree S — ML Data Operations Specialist**

A Python automation tool that checks ML annotation quality —
detecting inconsistencies, calculating inter-annotator agreement
(Cohen's Kappa), and generating quality reports.

## What This Project Does
- Detects missing annotations
- Checks label consistency across 3 annotators
- Calculates Cohen's Kappa score (achieved 0.82 — above 0.80 benchmark)
- Flags low confidence and slow annotations
- Generates CSV quality reports
- Creates visual quality dashboard

## Results
- Average Kappa Score: 0.82 (Almost Perfect Agreement)
- Quality Rate: 60% good quality records
- 8 records flagged for review out of 20

## Technologies Used
Python | Pandas | Scikit-learn | Matplotlib | Seaborn | Google Colab

## How to Run
1. Install libraries: pip install -r requirements.txt
2. Run checker: python src/quality_checker.py
3. Check output folder for reports and charts

## Project Structure
- src/quality_checker.py — main Python script
- data/sample_annotations.csv — annotation dataset
- quality_report_full.csv — complete quality report
- records_needing_review.csv — flagged records
- quality_dashboard.png — visual dashboard

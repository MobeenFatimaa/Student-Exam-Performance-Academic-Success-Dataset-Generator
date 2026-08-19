# Student Exam Performance & Academic Success Dataset Generator

A production-ready Python framework designed to generate a realistic synthetic dataset of **100,000 student academic records** across **44 features**.

The dataset is calibrated using non-linear physics and standardized normalization to simulate real-world distributions, academic grade curves, missing value patterns, and behavioral feature correlations.

**Author:** Mobeen Fatima

**Kaggle Dataset:** [Student Exam Performance & Academic Success Dataset](https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-academic-success-dataset)

---

## Dataset Highlights

* **Sample Size:** 100,000 students (`STU_000001` to `STU_100000`).
* **Feature Scope:** 44 total columns across 6 key domains (Demographics, Academics, Study Habits, Lifestyle, Technology, and Exam Factors).
* **Multi-Target Support:**
  * **Regression:** `exam_score` (Continuous score: 0–100)
  * **Binary Classification:** `pass_status` (Pass/Fail at threshold 50)
  * **Multi-class Classification:** `performance_grade` (A, B, C, D, F) and `performance_level` (High, Medium, Low)
* **Realism Engineering:**
  * Approximately 77% Pass Rate (within the standard 70–85% academic benchmark).
  * Balanced bell-curve grade distribution.
  * Controlled 5–10% missing value injection in selected features for imputation practice.
  * Realistic multi-factor correlations between effort, sleep, prior scores, anxiety, stress, and academic performance.

---

## Repository Structure

```text
.
├── generate_dataset.py    # Generates raw synthetic data and exports student_exam_performance.csv
├── validate_dataset.py    # Loads exported dataset and prints validation diagnostics
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rule file
└── README.md              # Project documentation

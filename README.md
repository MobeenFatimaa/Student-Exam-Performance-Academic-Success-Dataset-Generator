# Student Exam Performance & Academic Success Dataset Generator

A production-ready Python framework designed to generate a realistic synthetic dataset of **100,000 student academic records** across **44 features**.

The dataset is calibrated using non-linear physics and standardized normalization to simulate real-world distributions, academic grade curves, missing value patterns, and behavioral feature correlations.

---

## Dataset Highlights

* **Sample Size:** 100,000 students (`STU_000001` to `STU_100000`)
* **Feature Scope:** 44 total columns across 6 key domains: Demographics, Academics, Study Habits, Lifestyle, Technology, and Exam Factors
* **Multi-Target Support:**

  * **Regression:** `exam_score` — Continuous score from 0 to 100
  * **Binary Classification:** `pass_status` — Pass/Fail at a threshold of 50
  * **Multi-class Classification:** `performance_grade` — A, B, C, D, F
  * **Multi-class Classification:** `performance_level` — High, Medium, Low
* **Realism Engineering:**

  * Approximately 77% pass rate
  * Balanced academic grade distribution
  * Controlled 5–10% missing value injection in selected features for imputation practice
  * Realistic multi-factor correlations between effort, sleep, previous scores, anxiety, stress, and academic performance

---

## Kaggle Dataset

The complete generated dataset is available on Kaggle:

**Kaggle Dataset:**
https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-and-success-dataset

Example:

`https://www.kaggle.com/datasets/mobeenfatimah/student-exam-performance-academic-success`

---

## Repository Structure

```text
.
├── generate_dataset.py    # Generates raw synthetic data and exports student_exam_performance.csv
├── validate_dataset.py    # Loads exported dataset and prints validation diagnostics
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rule file
└── README.md              # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MobeenFatimaa/Student-Exam-Performance-Academic-Success-Dataset-Generator.git
cd Student-Exam-Performance-Academic-Success-Dataset-Generator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset

Run the generation script to create `student_exam_performance.csv` containing 100,000 rows and 44 columns:

```bash
python generate_dataset.py
```

### 4. Validate the Dataset

Execute the validation suite to inspect primary keys, target distributions, grade curves, missingness counts, and feature correlations:

```bash
python validate_dataset.py
```

---

## Feature Breakdown

| Domain                        | Key Features                                                                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Demographics**              | `age`, `gender`, `education_level`, `school_type`, `family_income`, `parent_education`, `urban_rural`                                                                                                                           |
| **Academics**                 | `previous_exam_score`, `previous_gpa`, `attendance_percentage`, `assignment_completion_rate`, `class_participation`, `private_tuition`                                                                                          |
| **Study Habits**              | `study_hours_per_day`, `self_study_hours`, `study_consistency`, `study_environment`, `study_method`, `revision_frequency`, `practice_tests_completed`, `notes_quality`                                                          |
| **Lifestyle**                 | `sleep_hours`, `sleep_quality`, `daily_screen_time`, `physical_activity_hours`, `break_frequency`, `stress_level`, `motivation_level`                                                                                           |
| **Technology & Exam Factors** | `internet_access`, `device_availability`, `educational_app_usage`, `online_course_hours`, `exam_difficulty`, `exam_preparation_days`, `questions_attempted`, `questions_correct`, `time_management_score`, `exam_anxiety_level` |
| **Targets**                   | `exam_score`, `performance_grade`, `pass_status`, `performance_level`                                                                                                                                                           |

---

## Machine Learning Use Cases

This dataset is designed for a wide range of machine learning and data science applications:

1. **Exploratory Data Analysis (EDA)**
   Analyze relationships between student demographics, lifestyle, study habits, and academic outcomes.

2. **Regression Modeling**
   Predict continuous `exam_score` values based on academic, behavioral, lifestyle, and examination factors.

3. **Binary Classification**
   Predict `pass_status` to identify whether a student is likely to pass or fail an examination.

4. **Multi-class Classification**
   Predict `performance_grade` and `performance_level` using student characteristics and behavioral indicators.

5. **Missing Value Imputation**
   Practice handling controlled missingness across features such as `previous_gpa`, `attendance_percentage`, and `time_management_score`.

6. **Feature Importance Analysis**
   Quantify the relative contribution of factors such as `study_hours_per_day`, `previous_exam_score`, `stress_level`, `motivation_level`, and `exam_anxiety_level`.

7. **At-Risk Student Identification**
   Develop predictive models for identifying students who may be at risk of poor academic performance.

8. **Educational Analytics**
   Explore how study habits, technology access, lifestyle patterns, and socioeconomic factors relate to academic success.

9. **Model Benchmarking**
   Compare algorithms such as Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Support Vector Machines, and neural networks.

10. **Data Preprocessing and Feature Engineering**
    Practice categorical encoding, numerical scaling, missing-value treatment, feature selection, and pipeline construction.

---

## Target Variables

### `exam_score`

A continuous numerical target representing the student's final examination score on a scale from 0 to 100.

**Task:** Regression

### `pass_status`

A binary target representing examination outcome.

* `Pass` — Exam score >= 50
* `Fail` — Exam score < 50

**Task:** Binary Classification

### `performance_grade`

A categorical academic grade derived from the examination score.

Possible values:

* `A`
* `B`
* `C`
* `D`
* `F`

**Task:** Multi-class Classification

### `performance_level`

A broader performance category representing the student's overall academic achievement.

Possible values:

* `High`
* `Medium`
* `Low`

**Task:** Multi-class Classification

---

## Dataset Characteristics

The generator is designed to create synthetic records with realistic relationships rather than independently sampled random values.

Examples of modeled relationships include:

* Higher study time generally contributes to higher exam performance.
* Previous academic performance influences current examination outcomes.
* Attendance contributes positively to academic achievement.
* Better sleep quality can support stronger academic performance.
* Higher motivation is associated with greater study consistency.
* Excessive stress and exam anxiety can negatively influence performance.
* Access to technology and educational resources can influence study effectiveness.
* Practice tests and revision frequency contribute to examination preparedness.
* Time management influences the relationship between preparation and final performance.

The dataset is entirely synthetic and does not represent real students, schools, universities, or institutions.

---

## Data Quality and Validation

The included `validate_dataset.py` script provides automated diagnostics for:

* Dataset dimensions
* Unique student IDs
* Duplicate records
* Missing values
* Target variable distributions
* Pass/fail distribution
* Performance grade distribution
* Performance level distribution
* Numerical feature ranges
* Feature correlations
* Basic dataset integrity checks

This makes the repository suitable for experimenting with complete machine learning workflows from data generation through validation and modeling.

---

## Reproducibility

The dataset generation process is designed to be reproducible through controlled randomization and deterministic generation logic.

To regenerate the dataset, run:

```bash
python generate_dataset.py
```

The generated CSV file can then be validated using:

```bash
python validate_dataset.py
```

---

## Technologies Used

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Statistical Distribution Modeling
* Synthetic Data Generation

---

## Intended Applications

This dataset can be used for:

* Machine learning education
* Academic performance prediction
* Educational data mining
* Data science portfolio projects
* Classification benchmarking
* Regression benchmarking
* Exploratory data analysis
* Missing data experiments
* Feature engineering
* Model explainability
* Student risk prediction research
* Synthetic data generation research

---

## Limitations

This dataset is synthetically generated and should not be interpreted as a representation of actual student populations.

The relationships and distributions are designed for realistic machine learning experimentation and educational analytics. They should not be used to make real-world decisions about individual students.

The dataset does not contain personally identifiable information.

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Author

**Mobeen Fatima**

GitHub:
https://github.com/MobeenFatimaa

---

## Project Links

**GitHub Repository:**
https://github.com/MobeenFatimaa/Student-Exam-Performance-Academic-Success-Dataset-Generator

**Kaggle Dataset:**
Replace the placeholder above with your actual published Kaggle dataset URL.

---

## Acknowledgments

This project was created as a synthetic data generation and machine learning benchmarking resource for students, researchers, educators, and data science practitioners.

If you find this dataset useful, consider giving the repository a star and sharing the dataset with other machine learning practitioners.

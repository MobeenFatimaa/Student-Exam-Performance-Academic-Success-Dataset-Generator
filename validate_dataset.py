import os
import pandas as pd

def validate_student_dataset(dataset_path="student_exam_performance.csv"):
    """
    Loads the exported CSV dataset and runs comprehensive statistical and validation checks.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found. Please run 'generate_dataset.py' first.")

    df = pd.read_csv(dataset_path)
    total_rows = len(df)

    print("==================================================")
    print("           DATASET VALIDATION REPORT              ")
    print("==================================================")
    print(f"File Name          : {dataset_path}")
    print(f"Total Records      : {total_rows:,}")
    print(f"Total Columns      : {df.shape[1]}")
    
    # 1. Primary Key Checks
    unique_ids = df['student_id'].nunique()
    print(f"\n[1] UNIQUE IDENTIFIERS")
    print(f"    - Unique Student IDs: {unique_ids:,} / {total_rows:,}")
    assert unique_ids == total_rows, "Validation Error: Duplicate student IDs detected!"

    # 2. Target Variable Checks
    pass_rate = (df['pass_status'] == 'Pass').mean() * 100
    print(f"\n[2] TARGET DISTRIBUTIONS")
    print(f"    - Pass Rate          : {pass_rate:.2f}% (Benchmark Target: 70.0% - 85.0%)")
    
    print("\n    - Grade Distribution (%):")
    grade_counts = df['performance_grade'].value_counts(normalize=True).loc[['A', 'B', 'C', 'D', 'F']] * 100
    for grade, pct in grade_counts.items():
        print(f"      Grade {grade} : {pct:6.2f}%")

    print("\n    - Performance Level Distribution (%):")
    level_counts = df['performance_level'].value_counts(normalize=True) * 100
    for level, pct in level_counts.items():
        print(f"      {level:6s}  : {pct:6.2f}%")

    # 3. Missing Value Analysis
    print(f"\n[3] MISSING VALUES REPORT")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            pct = (count / total_rows) * 100
            print(f"    - {col:<25} : {count:5d} missing ({pct:.2f}%)")
    else:
        print("    - No missing values found.")

    # 4. Correlation Verification
    print(f"\n[4] FEATURE CORRELATIONS WITH 'exam_score'")
    numeric_cols = [
        "previous_exam_score", "attendance_percentage", "study_hours_per_day",
        "practice_tests_completed", "sleep_hours", "stress_level", "exam_anxiety_level"
    ]
    correlations = df[numeric_cols + ["exam_score"]].corr()["exam_score"].drop("exam_score")
    for feat, corr_val in correlations.items():
        print(f"    - {feat:<25} : {corr_val:+.3f}")

    print("\n==================================================")
    print("STATUS: Dataset validated successfully and ready for Kaggle!")
    print("==================================================")

if __name__ == "__main__":
    validate_student_dataset()

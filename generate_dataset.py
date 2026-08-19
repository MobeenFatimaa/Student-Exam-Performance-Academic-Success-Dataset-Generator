import numpy as np
import pandas as pd

# Set reproducible seed
np.random.seed(42)
N = 100_000

print(f"Generating synthetic dataset for {N:,} students...")

# -----------------------------------------------------------------------------
# 1. DEMOGRAPHICS & BACKGROUND
# -----------------------------------------------------------------------------
student_id = [f"STU_{i:06d}" for i in range(1, N + 1)]
age = np.random.randint(14, 21, size=N)
gender = np.random.choice(["Male", "Female", "Other"], size=N, p=[0.49, 0.49, 0.02])
education_level = np.random.choice(["High School", "Undergraduate"], size=N, p=[0.6, 0.4])
school_type = np.random.choice(["Public", "Private", "Charter"], size=N, p=[0.55, 0.35, 0.10])
urban_rural = np.random.choice(["Urban", "Suburban", "Rural"], size=N, p=[0.45, 0.35, 0.20])

family_income = np.random.choice(
    ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
    size=N,
    p=[0.20, 0.25, 0.30, 0.15, 0.10]
)

parent_education = np.random.choice(
    ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    size=N,
    p=[0.30, 0.20, 0.30, 0.15, 0.05]
)

# Latent Socioeconomic Status (SES) factor to drive realistic correlations
ses_map = {"Low": -1.2, "Lower-Middle": -0.5, "Middle": 0.0, "Upper-Middle": 0.6, "High": 1.2}
ses_score = np.array([ses_map[i] for i in family_income]) + np.random.normal(0, 0.3, N)

# -----------------------------------------------------------------------------
# 2. ACADEMICS & STUDY BEHAVIOR
# -----------------------------------------------------------------------------
# Base academic baseline driven slightly by SES + noise
base_ability = np.random.normal(70, 10, N) + (ses_score * 3)
previous_exam_score = np.clip(base_ability + np.random.normal(0, 5, N), 30, 100)
previous_gpa = np.clip((previous_exam_score / 25) + np.random.normal(0, 0.15, N), 1.0, 4.0)

attendance_percentage = np.clip(75 + (ses_score * 2) + np.random.normal(10, 8, N), 40, 100)
study_hours_per_day = np.clip(np.random.gamma(shape=3.0, scale=1.0, size=N), 0.5, 12.0)
self_study_hours = np.clip(study_hours_per_day * np.random.uniform(0.5, 0.9, size=N), 0.2, 10.0)

assignment_completion_rate = np.clip(
    (attendance_percentage * 0.6) + (study_hours_per_day * 3) + np.random.normal(10, 10, N), 20, 100
)

class_participation = np.random.choice(["Low", "Medium", "High"], size=N, p=[0.25, 0.50, 0.25])
private_tuition = np.random.choice([0, 1], size=N, p=[0.7, 0.3])
online_learning_hours = np.clip(np.random.exponential(scale=2.0, size=N), 0, 15)

study_consistency = np.random.choice(["Low", "Medium", "High"], size=N, p=[0.20, 0.55, 0.25])
study_environment = np.random.choice(["Quiet", "Moderate", "Noisy"], size=N, p=[0.50, 0.35, 0.15])
study_method = np.random.choice(["Flashcards", "Group Study", "Practice Tests", "Self-Reading", "Summarizing"], size=N)
revision_frequency = np.random.choice(["Rarely", "Weekly", "Daily"], size=N, p=[0.20, 0.50, 0.30])
practice_tests_completed = np.random.poisson(lam=4 + (study_hours_per_day * 0.5), size=N)
notes_quality = np.random.choice(["Poor", "Average", "Excellent"], size=N, p=[0.20, 0.60, 0.20])

# -----------------------------------------------------------------------------
# 3. LIFESTYLE, TECH & EXAM FACTORS
# -----------------------------------------------------------------------------
sleep_hours = np.clip(np.random.normal(7, 1.2, N), 3, 11)
sleep_quality = np.random.choice(["Poor", "Fair", "Good", "Excellent"], size=N, p=[0.15, 0.35, 0.35, 0.15])
daily_screen_time = np.clip(np.random.normal(5, 1.8, N), 1, 14)
physical_activity_hours = np.clip(np.random.exponential(scale=2.5, size=N), 0, 12)
break_frequency = np.random.choice(["Rarely", "Occasionally", "Frequently"], size=N, p=[0.2, 0.5, 0.3])

motivation_level = np.random.choice(["Low", "Medium", "High"], size=N, p=[0.2, 0.5, 0.3])
stress_level = np.clip(np.random.randint(1, 11, N) + (10 - sleep_hours * 0.5), 1, 10).astype(int)

internet_access = np.random.choice([1, 0], size=N, p=[0.92, 0.08])
device_availability = np.random.choice(["Shared", "Dedicated", "None"], size=N, p=[0.25, 0.72, 0.03])
educational_app_usage = np.random.choice(["Low", "Moderate", "High"], size=N, p=[0.4, 0.4, 0.2])
online_course_hours = np.clip(np.random.exponential(scale=3.0, size=N), 0, 20)

exam_difficulty = np.random.choice(["Easy", "Medium", "Hard"], size=N, p=[0.25, 0.50, 0.25])
exam_preparation_days = np.random.randint(1, 31, size=N)
exam_anxiety_level = np.clip(stress_level + np.random.normal(0, 1.5, N), 1, 10)
time_management_score = np.clip(np.random.normal(65, 15, N) + (sleep_hours * 2), 10, 100)

# -----------------------------------------------------------------------------
# 4. TARGET GENERATION (Calibrated Physics Engine)
# -----------------------------------------------------------------------------
diff_penalty = {"Easy": 6, "Medium": 0, "Hard": -6}
exam_diff_numeric = np.array([diff_penalty[d] for d in exam_difficulty])

# Controlled interaction term
anxiety_prep_interaction = -1 * (exam_anxiety_level / (exam_preparation_days + 1)) * 3

# Calibrated score calculation with a +22.0 baseline intercept adjustment
raw_score = (
    22.0 +  # Intercept bump to push mean score into realistic range (~68)
    0.35 * previous_exam_score +
    0.20 * attendance_percentage +
    1.80 * (study_hours_per_day ** 0.85) +
    0.75 * practice_tests_completed +
    0.05 * time_management_score +
    1.00 * (sleep_hours - 7) +
    -1.00 * stress_level +
    exam_diff_numeric +
    anxiety_prep_interaction +
    np.random.normal(0, 5.5, N)  # Noise variance
)

# Rescale and clip exam scores between 0 and 100
exam_score = np.clip(np.round(raw_score, 2), 0, 100)

# Derive question performance from score
questions_attempted = np.random.randint(85, 101, size=N)
questions_correct = np.clip(
    np.round((exam_score / 100) * questions_attempted + np.random.normal(0, 1.5, N)),
    0,
    questions_attempted
).astype(int)

# Target Derivations
pass_status = np.where(exam_score >= 50, "Pass", "Fail")

def get_grade(s):
    if s >= 85: return "A"
    elif s >= 75: return "B"
    elif s >= 65: return "C"
    elif s >= 50: return "D"
    else: return "F"

performance_grade = pd.Series(exam_score).apply(get_grade)

def get_level(s):
    if s >= 80: return "High"
    elif s >= 60: return "Medium"
    else: return "Low"

performance_level = pd.Series(exam_score).apply(get_level)

# -----------------------------------------------------------------------------
# 5. ASSEMBLE DATAFRAME & INJECT MISSING VALUES
# -----------------------------------------------------------------------------
df = pd.DataFrame({
    "student_id": student_id,
    "age": age,
    "gender": gender,
    "education_level": education_level,
    "school_type": school_type,
    "family_income": family_income,
    "parent_education": parent_education,
    "urban_rural": urban_rural,
    "previous_exam_score": previous_exam_score.round(2),
    "previous_gpa": previous_gpa.round(2),
    "attendance_percentage": attendance_percentage.round(2),
    "assignment_completion_rate": assignment_completion_rate.round(2),
    "class_participation": class_participation,
    "study_hours_per_day": study_hours_per_day.round(2),
    "self_study_hours": self_study_hours.round(2),
    "private_tuition": private_tuition,
    "online_learning_hours": online_learning_hours.round(2),
    "study_consistency": study_consistency,
    "study_environment": study_environment,
    "study_method": study_method,
    "revision_frequency": revision_frequency,
    "practice_tests_completed": practice_tests_completed,
    "notes_quality": notes_quality,
    "sleep_hours": sleep_hours.round(2),
    "sleep_quality": sleep_quality,
    "daily_screen_time": daily_screen_time.round(2),
    "physical_activity_hours": physical_activity_hours.round(2),
    "break_frequency": break_frequency,
    "stress_level": stress_level,
    "motivation_level": motivation_level,
    "internet_access": internet_access,
    "device_availability": device_availability,
    "educational_app_usage": educational_app_usage,
    "online_course_hours": online_course_hours.round(2),
    "exam_difficulty": exam_difficulty,
    "exam_preparation_days": exam_preparation_days,
    "questions_attempted": questions_attempted,
    "questions_correct": questions_correct,
    "time_management_score": time_management_score.round(2),
    "exam_anxiety_level": exam_anxiety_level.round(2),
    "exam_score": exam_score,
    "performance_grade": performance_grade,
    "pass_status": pass_status,
    "performance_level": performance_level
})

# Inject 5-10% missing values in selected columns for imputation projects
cols_to_impute = [
    "previous_gpa", "attendance_percentage", "parent_education",
    "sleep_quality", "time_management_score", "notes_quality"
]

for col in cols_to_impute:
    mask = np.random.rand(N) < np.random.uniform(0.05, 0.10)
    df.loc[mask, col] = np.nan

# -----------------------------------------------------------------------------
# 6. EXPORT & VALIDATION CHECKS
# -----------------------------------------------------------------------------
filename = "student_exam_performance.csv"
df.to_csv(filename, index=False)
print(f"Dataset saved successfully as '{filename}' ({df.shape[0]} rows, {df.shape[1]} columns).\n")

# Run validation diagnostics
print("=== VALIDATION REPORT ===")
print(f"Unique Student IDs : {df['student_id'].nunique()} / {N}")
print(f"Pass Rate          : {(df['pass_status'] == 'Pass').mean() * 100:.2f}% (Target: 70-85%)")
print("\nGrade Distribution:")
print(df['performance_grade'].value_counts(normalize=True).round(4) * 100)

print("\nMissing Values Count per Column:")
print(df[cols_to_impute].isnull().sum())

print("\nKey Feature Correlations with 'exam_score':")
numeric_cols = [
    "previous_exam_score", "attendance_percentage", "study_hours_per_day",
    "practice_tests_completed", "sleep_hours", "stress_level", "exam_anxiety_level"
]
print(df[numeric_cols + ["exam_score"]].corr()["exam_score"].drop("exam_score").round(3))

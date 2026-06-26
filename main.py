import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor


class MarkPredictor:
    """Lightweight linear regression wrapper for compatibility tests."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class AcademicPredictor:
    """A multi-feature predictor with automated cleaning, scaling, and evaluation."""

    def __init__(self, feature_columns=None, target_column='Software_Engineering_Final'):
        self.feature_columns = feature_columns or ['Maths_Advanced', 'Physics', 'Modern_History']
        self.target_column = target_column
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression()),
        ])

    def load_data(self, path='master_markbook.csv'):
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        df.columns = [col.strip() for col in df.columns]
        return df

    def clean_data(self, df):
        required_columns = ['Student_Name'] + self.feature_columns + [self.target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')

        df_clean = df.copy()
        numeric_columns = self.feature_columns + [self.target_column]
        for col in numeric_columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        df_clean = df_clean.dropna(subset=required_columns)
        for col in numeric_columns:
            df_clean = df_clean[df_clean[col].between(0, 100)]

        df_clean = df_clean.reset_index(drop=True)
        return df_clean

    def save_cleaned_data(self, df, path='master_markbook_cleaned.csv'):
        df.to_csv(path, index=False)
        return path

    def prepare_data(self, df, test_size=0.2, random_state=42):
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_rmse(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        return rmse, predictions

    def cross_validate(self, df, cv=5):
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        return -cv_scores.mean()

    def predict_student(self, student_row):
        if student_row is None:
            return None
        X_student = student_row[self.feature_columns].to_numpy().reshape(1, -1)
        return float(self.model.predict(X_student)[0])


def check_data_reliability(attendance_percentage):
    if attendance_percentage < 50.0:
        print('ACCESS DENIED')
        return False
    print('ACCESS GRANTED')
    return True


def load_and_clean_data(path='master_markbook.csv'):
    predictor = AcademicPredictor()
    df_raw = predictor.load_data(path)
    df_clean = predictor.clean_data(df_raw)
    print(f'\ndf_raw rows: {len(df_raw)}')
    print(f'df_clean rows: {len(df_clean)}')
    return df_raw, df_clean


def train_simple_linear_model(df_clean):
    X1 = df_clean[['Maths_Advanced']].values
    y = df_clean['Software_Engineering_Final'].values

    X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X1_train, y_train)

    y_pred = model.predict(X1_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'\nBaseline RMSE: {rmse:.2f}')

    weight = model.coef_[0]
    bias = model.intercept_
    print(f'Baseline equation: Software_Engineering_Final = {weight:.2f} * Maths_Advanced + {bias:.2f}')

    return model, X1_train, X1_test, y_train, y_test, y_pred


def train_mark_predictor(X_train, y_train, X_test, y_test):
    my_ai = MarkPredictor()
    my_ai.fit(X_train, y_train)
    y_pred_ai = my_ai.predict(X_test)
    rmse_ai = np.sqrt(mean_squared_error(y_test, y_pred_ai))
    print(f'\nRMSE from MarkPredictor class: {rmse_ai:.2f}')
    return my_ai, y_pred_ai, rmse_ai


def train_level2_ai(df_clean, y):
    print('--- Training Level 2 AI (Multi-Subject) ---')

    X2 = df_clean[['Maths_Advanced', 'Physics']].values
    X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train)
    X2_test_scaled = scaler.transform(X2_test)

    my_ai_level2 = MarkPredictor()
    my_ai_level2.fit(X2_train_scaled, y_train)
    y_pred_level2 = my_ai_level2.predict(X2_test_scaled)
    rmse_level2 = np.sqrt(mean_squared_error(y_test, y_pred_level2))
    print(f'Level 2 RMSE: {rmse_level2:.2f}')

    return my_ai_level2, scaler, X2, X2_train_scaled, X2_test_scaled, y_train, y_test, y_pred_level2, rmse_level2


def find_student_row(df, student_name='Alex Anderson'):
    matches = df[df['Student_Name'].str.contains(student_name, case=False, na=False)]
    if matches.empty:
        print(f"\nNo student found with name matching '{student_name}'.")
        return None
    student_row = matches.iloc[0]
    print(f"\nFound student record for {student_row['Student_Name']}")
    return student_row


def predict_student_final_score(student_row, scaler, model):
    if student_row is None:
        return None

    feature_cols = ['Maths_Advanced', 'Physics']
    X_student = student_row[feature_cols].values.reshape(1, -1)
    X_student_scaled = scaler.transform(X_student)
    prediction = model.predict(X_student_scaled)[0]
    print(f"\nPredicted Software_Engineering_Final for {student_row['Student_Name']}: {prediction:.1f}")
    return prediction


def save_prediction_output(student_name, predicted_score, model_name='Level 2 AI', path='alex_prediction.csv'):
    output = pd.DataFrame([
        {
            'Student_Name': student_name,
            'Model': model_name,
            'Predicted_Software_Engineering_Final': round(predicted_score, 1),
        }
    ])
    output.to_csv(path, index=False)
    print(f'\nPrediction output saved to {path}')
    return path


def run_bias_audit(df_clean):
    print('--- Running Bias Audit (80% Rule) ---')
    # Default groups: compare students strong in Physics vs Modern History
    group_a = df_clean[df_clean['Physics'] > 70]
    group_b = df_clean[df_clean['Modern_History'] > 70]

    # Calculate pass rates (>=50)
    pass_rate_a = group_a['Software_Engineering_Final'].ge(50).mean() if len(group_a) > 0 else 0.0
    pass_rate_b = group_b['Software_Engineering_Final'].ge(50).mean() if len(group_b) > 0 else 0.0

    # Handle zero divisions and small sample sizes
    disparate_impact_ratio = pass_rate_b / pass_rate_a if pass_rate_a not in (0, None) else 0.0

    # Provide warnings for small groups
    warnings = []
    if len(group_a) < 5 or len(group_b) < 5:
        warnings.append('Small sample size: results may be unreliable')

    if disparate_impact_ratio < 0.8 and disparate_impact_ratio > 0:
        warnings.append(f'Bias suspected: disparate impact ratio = {disparate_impact_ratio:.2f}')

    if disparate_impact_ratio == 0 and pass_rate_a == 0 and pass_rate_b == 0:
        warnings.append('No passes in either group')

    if warnings:
        print('\nWARNING: ' + '; '.join(warnings))
    else:
        print(f'\nAudit passed. Disparate Impact Ratio = {disparate_impact_ratio:.2f}')

    return disparate_impact_ratio


def bias_audit_report(df_clean, group_a_filter=None, group_b_filter=None, pass_threshold=50, min_group_size=5):
    """Return a detailed bias audit report dict for the provided filters.

    group_a_filter/group_b_filter are callables that accept the dataframe and
    return boolean masks. If not provided, defaults mirror run_bias_audit.
    """
    if group_a_filter is None:
        group_a_filter = lambda df: df['Physics'] > 70
    if group_b_filter is None:
        group_b_filter = lambda df: df['Modern_History'] > 70

    group_a = df_clean[group_a_filter(df_clean)]
    group_b = df_clean[group_b_filter(df_clean)]

    pass_rate_a = group_a['Software_Engineering_Final'].ge(pass_threshold).mean() if len(group_a) > 0 else 0.0
    pass_rate_b = group_b['Software_Engineering_Final'].ge(pass_threshold).mean() if len(group_b) > 0 else 0.0

    disparate_impact_ratio = pass_rate_b / pass_rate_a if pass_rate_a not in (0, None) else 0.0

    report = {
        'group_a_size': int(len(group_a)),
        'group_b_size': int(len(group_b)),
        'pass_rate_a': float(pass_rate_a),
        'pass_rate_b': float(pass_rate_b),
        'disparate_impact_ratio': float(disparate_impact_ratio),
        'warnings': [],
    }

    if report['group_a_size'] < min_group_size or report['group_b_size'] < min_group_size:
        report['warnings'].append('Small sample size')

    if report['disparate_impact_ratio'] < 0.8:
        report['warnings'].append('Potential disparate impact (ratio < 0.8)')

    return report


def check_data_privacy(df, pii_columns=None):
    """Check dataframe for presence of PII columns and return findings."""
    pii_columns = pii_columns or ['Student_Name']
    found = [col for col in pii_columns if col in df.columns]
    result = {
        'pii_found': bool(found),
        'columns': found,
    }
    if result['pii_found']:
        print(f"PII columns detected: {found}. Consider anonymizing before sharing.")
    else:
        print('No PII columns detected')
    return result


def anonymize_and_save(df, path='master_markbook_anonymized.csv'):
    """Replace Student_Name with a deterministic hashed ID and save CSV."""
    if 'Student_Name' not in df.columns:
        raise ValueError('Student_Name column required for anonymization')

    df_out = df.copy()
    # Use SHA256 so IDs are stable across runs and environments.
    df_out['Student_ID'] = df_out['Student_Name'].astype(str).map(
        lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()[:12]
    )
    df_out = df_out.drop(columns=['Student_Name'])
    df_out.to_csv(path, index=False)
    print(f'Anonymized data saved to {path}')
    return path


def calculate_file_sha256(path):
    """Return SHA256 hash for a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def secure_load_clean_and_anonymize(raw_path='master_markbook.csv', anon_path='master_markbook_anonymized.csv'):
    """Load and clean data, anonymize before training, and verify raw-file integrity."""
    predictor = AcademicPredictor()
    raw_hash_before = calculate_file_sha256(raw_path)

    df_raw = predictor.load_data(raw_path)
    df_clean = predictor.clean_data(df_raw)

    privacy_result = check_data_privacy(df_clean)
    anonymized_path = anonymize_and_save(df_clean, path=anon_path)
    df_train = pd.read_csv(anonymized_path)

    raw_hash_after = calculate_file_sha256(raw_path)
    integrity_ok = raw_hash_before == raw_hash_after
    if not integrity_ok:
        raise RuntimeError('Raw data integrity check failed: source file changed during processing')

    print(f'Raw data integrity verified (SHA256 unchanged): {raw_hash_before[:12]}...')

    return {
        'df_raw': df_raw,
        'df_clean': df_clean,
        'df_train': df_train,
        'privacy_result': privacy_result,
        'anonymized_path': anonymized_path,
        'raw_hash_before': raw_hash_before,
        'raw_hash_after': raw_hash_after,
        'integrity_ok': integrity_ok,
    }


def log_rmse(model_name, rmse, path='rmse_report.csv'):
    """Append RMSE entry for a model to a CSV report file and return path."""
    import os
    import csv

    header = ['timestamp', 'model', 'rmse']
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([datetime.now(timezone.utc).isoformat(), model_name, float(rmse)])
    print(f'Logged RMSE for {model_name} -> {path}')
    return path


def cross_validation_check(X2, y, scaler):
    """Perform 5-fold cross-validation and return detailed metrics."""
    print('--- Cross-Validation Check ---')
    X2_scaled = scaler.fit_transform(X2)
    cv_scores = cross_val_score(LinearRegression(), X2_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse_scores = -cv_scores
    mean_cv_rmse = cv_rmse_scores.mean()
    std_cv_rmse = cv_rmse_scores.std()
    
    print(f'Mean Cross-Validation RMSE: {mean_cv_rmse:.4f} ± {std_cv_rmse:.4f}')
    print('\nFold-by-fold Results:')
    for fold_idx, rmse in enumerate(cv_rmse_scores, 1):
        print(f'  Fold {fold_idx}: {rmse:.4f}')
    print(f'\nCoefficient of Variation: {(std_cv_rmse/mean_cv_rmse)*100:.2f}%')
    
    return mean_cv_rmse, cv_rmse_scores


def detailed_cross_validation_report(X2, y, scaler):
    """Generate comprehensive cross-validation report with overfitting analysis."""
    from sklearn.model_selection import train_test_split
    
    print('\n=== CROSS-VALIDATION ROBUSTNESS REPORT ===\n')
    
    # Run cross-validation
    X2_scaled = scaler.fit_transform(X2)
    cv_scores = cross_val_score(LinearRegression(), X2_scaled, y, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse_scores = -cv_scores
    
    # Run train-test split for comparison
    X2_train, X2_test, y_train, y_test = train_test_split(X2_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X2_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X2_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X2_test)))
    
    # Compile report
    report = {
        'cv_mean_rmse': float(cv_rmse_scores.mean()),
        'cv_std_rmse': float(cv_rmse_scores.std()),
        'cv_fold_scores': [float(score) for score in cv_rmse_scores],
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'overfitting_gap': float(test_rmse - train_rmse),
        'overfitting_ratio': float(test_rmse / train_rmse) if train_rmse > 0 else 0.0,
        'cv_coefficient_of_variation': float((cv_rmse_scores.std() / cv_rmse_scores.mean()) * 100),
    }
    
    # Print formatted report
    print('Cross-Validation Metrics:')
    print(f'  Mean CV RMSE: {report["cv_mean_rmse"]:.4f}')
    print(f'  Std Dev: {report["cv_std_rmse"]:.4f}')
    print(f'  Coefficient of Variation: {report["cv_coefficient_of_variation"]:.2f}%')
    
    print('\nFold-by-fold RMSE:')
    for fold_idx, rmse in enumerate(report['cv_fold_scores'], 1):
        print(f'  Fold {fold_idx}: {rmse:.4f}')
    
    print('\nTrain-Test Split Comparison:')
    print(f'  Training RMSE: {report["train_rmse"]:.4f}')
    print(f'  Testing RMSE: {report["test_rmse"]:.4f}')
    print(f'  Overfitting Gap: {report["overfitting_gap"]:.4f}')
    print(f'  Overfitting Ratio (Test/Train): {report["overfitting_ratio"]:.4f}')
    
    print('\nRobustness Assessment:')
    if report["overfitting_ratio"] < 1.5:
        print('  ✅ EXCELLENT: Minimal overfitting detected')
    elif report["overfitting_ratio"] < 2.0:
        print('  ✅ GOOD: Acceptable overfitting level (expected with small datasets)')
    else:
        print('  ⚠️  MODERATE: Some overfitting present, but mitigated by scaling and simple model')
    
    return report


def neural_network_test(X2_train_scaled, X2_test_scaled, y_train, y_test, rmse_level2):
    print('--- Extension: Neural Network Test ---')
    nn_model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1500, random_state=42)
    nn_model.fit(X2_train_scaled, y_train)

    y_pred_nn = nn_model.predict(X2_test_scaled)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    print(f'\nLevel 2 Linear RMSE: {rmse_level2:.2f}')
    print(f'Neural Network RMSE: {rmse_nn:.2f}')
    return nn_model, y_pred_nn, rmse_nn


def main():
    print('\n✅ All libraries successfully imported!')
    prep = secure_load_clean_and_anonymize()
    df_raw = prep['df_raw']
    df_clean = prep['df_clean']
    df_train = prep['df_train']

    print(f'\ndf_raw rows: {len(df_raw)}')
    print(f'df_clean rows: {len(df_clean)}')
    print(f'df_train rows (anonymized): {len(df_train)}')

    model, X1_train, X1_test, y_train, y_test, y_pred = train_simple_linear_model(df_train)
    log_rmse('Baseline', np.sqrt(mean_squared_error(y_test, y_pred)))
    my_ai, y_pred_ai, rmse_ai = train_mark_predictor(X1_train, y_train, X1_test, y_test)

    y = df_train['Software_Engineering_Final'].values
    my_ai_level2, scaler, X2, X2_train_scaled, X2_test_scaled, y_train_lvl2, y_test_lvl2, y_pred_level2, rmse_level2 = train_level2_ai(df_train, y)
    log_rmse('Level 2 AI', rmse_level2)

    alex_row = find_student_row(df_raw, 'Alex')
    alex_prediction = predict_student_final_score(alex_row, scaler, my_ai_level2)
    if alex_prediction is not None:
        save_prediction_output(alex_row['Student_Name'], alex_prediction)

    run_bias_audit(df_train)
    
    # Enhanced cross-validation report
    detailed_cross_validation_report(X2, y, scaler)
    
    neural_network_test(X2_train_scaled, X2_test_scaled, y_train_lvl2, y_test_lvl2, rmse_level2)


if __name__ == '__main__':
    main()

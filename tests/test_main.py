import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from main import (
    AcademicPredictor,
    MarkPredictor,
    check_data_reliability,
    load_and_clean_data,
    train_simple_linear_model,
    train_mark_predictor,
    train_level2_ai,
    find_student_row,
    predict_student_final_score,
    save_prediction_output,
    run_bias_audit,
    bias_audit_report,
    check_data_privacy,
    anonymize_and_save,
    cross_validation_check,
    neural_network_test,
)


def test_check_data_reliability():
    assert not check_data_reliability(10.0)
    assert check_data_reliability(90.0)


def test_find_student_row(sample_dataframe):
    student = find_student_row(sample_dataframe, 'alex')
    assert student is not None
    assert student['Student_Name'] == 'Alex Anderson'


def test_predict_student_final_score(sample_dataframe):
    class DummyScaler:
        def transform(self, X):
            return X

    class DummyModel:
        def predict(self, X):
            return np.array([90.0])

    student_row = sample_dataframe.iloc[0]
    prediction = predict_student_final_score(student_row, DummyScaler(), DummyModel())
    assert prediction == 90.0


def test_save_prediction_output(tmp_path):
    path = tmp_path / 'test_output.csv'
    result_path = save_prediction_output('Alex Anderson', 92.5, path=str(path))
    assert result_path == str(path)
    output_df = pd.read_csv(result_path)
    assert output_df.iloc[0]['Predicted_Software_Engineering_Final'] == 92.5


@given(st.floats(min_value=0, max_value=100))
def test_check_data_reliability_hypothesis(value):
    result = check_data_reliability(value)
    assert isinstance(result, bool)


def test_academic_predictor_clean_data(tmp_path):
    df = pd.DataFrame([
        {
            'Student_Name': 'Alex Anderson',
            'Maths_Advanced': 90,
            'Physics': 85,
            'Modern_History': 80,
            'Software_Engineering_Final': 92,
        },
        {
            'Student_Name': 'Invalid Student',
            'Maths_Advanced': 150,
            'Physics': 85,
            'Modern_History': 80,
            'Software_Engineering_Final': 92,
        },
        {
            'Student_Name': 'Missing Fields',
            'Maths_Advanced': None,
            'Physics': 70,
            'Modern_History': 75,
            'Software_Engineering_Final': 80,
        },
    ])

    predictor = AcademicPredictor()
    cleaned = predictor.clean_data(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Student_Name'] == 'Alex Anderson'

    out_path = tmp_path / 'cleaned.csv'
    saved_path = predictor.save_cleaned_data(cleaned, path=str(out_path))
    assert saved_path == str(out_path)
    loaded = pd.read_csv(saved_path)
    assert loaded['Student_Name'].iloc[0] == 'Alex Anderson'


def test_academic_predictor_train_evaluate():
    df = pd.DataFrame([
        {
            'Student_Name': 'Alex Anderson',
            'Maths_Advanced': 80,
            'Physics': 70,
            'Modern_History': 60,
            'Software_Engineering_Final': 85,
        },
        {
            'Student_Name': 'Jamie Smith',
            'Maths_Advanced': 60,
            'Physics': 65,
            'Modern_History': 70,
            'Software_Engineering_Final': 75,
        },
        {
            'Student_Name': 'Casey Lee',
            'Maths_Advanced': 70,
            'Physics': 80,
            'Modern_History': 65,
            'Software_Engineering_Final': 82,
        },
        {
            'Student_Name': 'Taylor Gray',
            'Maths_Advanced': 50,
            'Physics': 55,
            'Modern_History': 60,
            'Software_Engineering_Final': 68,
        },
    ])

    predictor = AcademicPredictor()
    cleaned = predictor.clean_data(df)
    X_train, X_test, y_train, y_test = predictor.prepare_data(cleaned, test_size=0.5, random_state=42)
    predictor.train(X_train, y_train)
    rmse, predictions = predictor.evaluate_rmse(X_test, y_test)

    assert rmse >= 0
    assert len(predictions) == len(y_test)


def test_academic_predictor_predict_student(sample_dataframe):
    predictor = AcademicPredictor()
    predictor.model.fit(
        sample_dataframe[['Maths_Advanced', 'Physics', 'Modern_History']].astype(float),
        sample_dataframe['Software_Engineering_Final'].astype(float),
    )
    predicted = predictor.predict_student(sample_dataframe.iloc[0])
    assert isinstance(predicted, float)


def test_bandit_semgrep_placeholder():
    assert True


def test_run_bias_audit_sample(sample_dataframe):
    # Use a small sample to ensure function runs and returns a float
    ratio = run_bias_audit(sample_dataframe)
    assert isinstance(ratio, float)


def test_bias_audit_report(sample_dataframe):
    report = bias_audit_report(sample_dataframe)
    assert 'disparate_impact_ratio' in report
    assert isinstance(report['disparate_impact_ratio'], float)


def test_check_data_privacy_and_anonymize(tmp_path, sample_dataframe):
    # check_data_privacy should detect Student_Name
    res = check_data_privacy(sample_dataframe)
    assert res['pii_found'] is True

    out = tmp_path / 'anon.csv'
    path = anonymize_and_save(sample_dataframe, path=str(out))
    assert path == str(out)
    df_loaded = pd.read_csv(path)
    assert 'Student_ID' in df_loaded.columns
    assert 'Student_Name' not in df_loaded.columns

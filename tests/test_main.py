import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st

from main import (
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
    sample_df = sample_dataframe.copy()
    scaler = np.array([[1.0]])
    model = type('M', (), {'predict': lambda self, X: np.array([90.0])})()
    student_row = sample_df.iloc[0]
    prediction = predict_student_final_score(student_row, scaler, model)
    assert prediction == 90.0


def test_save_prediction_output(tmp_path):
    path = tmp_path / 'test_output.csv'
    result_path = save_prediction_output('Alex Anderson', 92.5, path=str(path))
    assert result_path == str(path)
    output_df = pd.read_csv(result_path)
    assert output_df.iloc[0]['Predicted_Software_Engineering_Final'] == 92.5


given(st.floats(min_value=0, max_value=100))
def test_check_data_reliability_hypothesis(value):
    result = check_data_reliability(value)
    assert isinstance(result, bool)


def test_bandit_semgrep_placeholder():
    assert True

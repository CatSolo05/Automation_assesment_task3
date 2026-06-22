import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


class MarkPredictor:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def check_data_reliability(attendance_percentage):
    if attendance_percentage < 50.0:
        print("ACCESS DENIED")
        return False
    print("ACCESS GRANTED")
    return True


def load_and_clean_data(path='master_markbook.csv'):
    df_raw = pd.read_csv(path, sep=',', encoding='latin-1')
    print(f"\ndf_raw rows: {len(df_raw)}")

    df_clean = df_raw.dropna()
    df_clean = df_clean[
        (df_clean['Maths_Advanced'].between(0, 100))
        & (df_clean['Physics'].between(0, 100))
        & (df_clean['Software_Engineering_Final'].between(0, 100))
    ]

    print(df_clean.head())
    print(f"df_clean rows: {len(df_clean)}")
    return df_raw, df_clean


def train_simple_linear_model(df_clean):
    X1 = df_clean[['Maths_Advanced']].values
    y = df_clean['Software_Engineering_Final'].values

    X1_train, X1_test, y_train, y_test = train_test_split(
        X1, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X1_train, y_train)

    y_pred = model.predict(X1_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nRMSE: {rmse:.2f}")

    weight = model.coef_[0]
    bias = model.intercept_
    print(f"\nFinal formula: Software_Engineering_Final = {weight:.2f} * Maths_Advanced + {bias:.2f}")

    plt.scatter(X1_test, y_test, color='blue', label='Actual')
    plt.plot(X1_test, y_pred, color='red', label='Predicted')
    plt.title('Maths_Advanced vs Software_Engineering_Final')
    plt.xlabel('Maths_Advanced')
    plt.ylabel('Software_Engineering_Final')
    plt.legend()
    plt.show()

    return model, X1_train, X1_test, y_train, y_test, y_pred


def train_mark_predictor(X_train, y_train, X_test, y_test):
    my_ai = MarkPredictor()
    my_ai.fit(X_train, y_train)
    y_pred_ai = my_ai.predict(X_test)
    rmse_ai = np.sqrt(mean_squared_error(y_test, y_pred_ai))
    print(f"\nRMSE from MarkPredictor class: {rmse_ai:.2f}")
    return my_ai, y_pred_ai, rmse_ai


def train_level2_ai(df_clean, y):
    print("--- Training Level 2 AI (Multi-Subject) ---")

    X2 = df_clean[['Maths_Advanced', 'Physics']].values
    X2_train, X2_test, y_train, y_test = train_test_split(
        X2, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X2_train_scaled = scaler.fit_transform(X2_train)
    X2_test_scaled = scaler.transform(X2_test)

    my_ai_level2 = MarkPredictor()
    my_ai_level2.fit(X2_train_scaled, y_train)

    y_pred_level2 = my_ai_level2.predict(X2_test_scaled)
    rmse_level2 = np.sqrt(mean_squared_error(y_test, y_pred_level2))
    print(f"\nLevel 2 RMSE: {rmse_level2:.2f}")

    return my_ai_level2, scaler, X2, X2_train_scaled, X2_test_scaled, y_train, y_test, y_pred_level2, rmse_level2


def find_student_row(df, student_name="Alex Anderson"):
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
    output = pd.DataFrame([{
        'Student_Name': student_name,
        'Model': model_name,
        'Predicted_Software_Engineering_Final': round(predicted_score, 1)
    }])
    output.to_csv(path, index=False)
    print(f"\nPrediction output saved to {path}")
    return path


def run_bias_audit(df_clean):
    print("--- Running Bias Audit (80% Rule) ---")
    group_a = df_clean[df_clean['Physics'] > 70]
    group_b = df_clean[df_clean['Modern_History'] > 70]

    pass_rate_a = group_a['Software_Engineering_Final'].ge(50).mean()
    pass_rate_b = group_b['Software_Engineering_Final'].ge(50).mean()
    disparate_impact_ratio = pass_rate_b / pass_rate_a if pass_rate_a != 0 else 0

    if disparate_impact_ratio < 0.8:
        print(f"\nWARNING: Bias detected! Disparate Impact Ratio = {disparate_impact_ratio:.2f}")
    else:
        print(f"\nAudit passed. Disparate Impact Ratio = {disparate_impact_ratio:.2f}")

    return disparate_impact_ratio


def cross_validation_check(X2, y, scaler):
    print("--- Cross-Validation Check ---")
    X2_scaled = scaler.fit_transform(X2)
    cv_scores = cross_val_score(
        LinearRegression(), X2_scaled, y, cv=5, scoring='neg_root_mean_squared_error'
    )
    mean_cv_rmse = -cv_scores.mean()
    print(f"Cross-Validation RMSE: {mean_cv_rmse:.2f}")
    return mean_cv_rmse


def neural_network_test(X2_train_scaled, X2_test_scaled, y_train, y_test, rmse_level2):
    print("--- Extension: Neural Network Test ---")
    nn_model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=1500, random_state=42)
    nn_model.fit(X2_train_scaled, y_train)

    y_pred_nn = nn_model.predict(X2_test_scaled)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    print(f"\nLevel 2 Linear RMSE: {rmse_level2:.2f}")
    print(f"Neural Network RMSE: {rmse_nn:.2f}")
    return nn_model, y_pred_nn, rmse_nn


def main():
    print("\n✅ All libraries successfully imported!")
    df_raw, df_clean = load_and_clean_data()
    model, X1_train, X1_test, y_train, y_test, y_pred = train_simple_linear_model(df_clean)

    my_ai, y_pred_ai, rmse_ai = train_mark_predictor(X1_train, y_train, X1_test, y_test)

    y = df_clean['Software_Engineering_Final'].values
    my_ai_level2, scaler, X2, X2_train_scaled, X2_test_scaled, y_train_lvl2, y_test_lvl2, y_pred_level2, rmse_level2 = train_level2_ai(df_clean, y)

    alex_row = find_student_row(df_raw, 'Alex')
    alex_prediction = predict_student_final_score(alex_row, scaler, my_ai_level2)
    if alex_prediction is not None:
        save_prediction_output(alex_row['Student_Name'], alex_prediction)

    run_bias_audit(df_clean)
    cross_validation_check(X2, y, scaler)

    print("--- Logic Gatekeeper ---")
    check_data_reliability(45.0)
    check_data_reliability(92.0)

    neural_network_test(X2_train_scaled, X2_test_scaled, y_train_lvl2, y_test_lvl2, rmse_level2)


if __name__ == '__main__':
    main()

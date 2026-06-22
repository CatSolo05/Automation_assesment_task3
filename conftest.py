import pytest


@pytest.fixture
def sample_dataframe():
    import pandas as pd

    return pd.DataFrame([
        {
            'Student_Name': 'Alex Anderson',
            'Maths_Advanced': 85,
            'Physics': 90,
            'Software_Engineering_Final': 88,
            'Modern_History': 75,
        },
        {
            'Student_Name': 'Jamie Smith',
            'Maths_Advanced': 40,
            'Physics': 35,
            'Software_Engineering_Final': 45,
            'Modern_History': 55,
        },
    ])

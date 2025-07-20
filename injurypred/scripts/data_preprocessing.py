import pandas as pd

def load_and_clean_data(filepath: str, target_column: str):
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath, engine='openpyxl')
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

    df.dropna(subset=[target_column], inplace=True)

    # Drop Stick Test if present
    if 'Stick Test (cm)' in df.columns:
        df.drop(columns=['Stick Test (cm)'], inplace=True)

    # Define the required features for the model
    required_features = [
        'Injury Duration (weeks)', 'Injury Occurred (weeks ago)', 'Trunk Flexion (cm)', 'BMI',
        'Weekly Training Hours', 'Current discomfort / Injury', 'Shoulder Flexion (deg)', 'Weight (kg)',
        'Coach', 'Coach exp', 'coaches success %',
        'Quad Circumference (cm)', 'Calf Circumference (cm)', 'Wrist Circumference (cm)',
        'Ankle Circumference (cm)', 'Upper Arm Circumference (cm)'
    ]

    # Filter only the columns that exist
    existing_features = [f for f in required_features if f in df.columns]
    df = df[existing_features + [target_column]]

    # Handle missing values
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    for column in non_numeric_columns:
        if not df[column].mode().empty:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Ensure 'Gender' column is handled if present
    if 'Gender' in df.columns:
        gender_mode = df['Gender'].mode()
        if not gender_mode.empty:
            df['Gender'] = df['Gender'].fillna(gender_mode[0])
        else:
            df['Gender'] = 'Unknown'

    # Drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)

    if df.isnull().sum().sum() > 0:
        print("Columns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        raise ValueError("Data still contains missing values after preprocessing.")

    # Amplify important features
    if 'Injury Duration (weeks)' in df.columns:
        df['Injury Duration (weeks)'] *= 2
    if 'Coach exp' in df.columns:
        df['Coach exp'] *= 1.5
    if 'coaches success %' in df.columns:
        df['coaches success %'] *= 2

    return df

def encode_categorical_features(df):
    from sklearn.preprocessing import LabelEncoder
    encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le
    return df, encoders

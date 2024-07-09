import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_path):
    """Load the dataset from CSV file."""
    return pd.read_csv(file_path)

def explore_data(data):
    """Explore the dataset."""
    print("First few rows:\n", data.head())
    print("\nDataset information:\n", data.info())
    print("\nMissing values:\n", data.isnull().sum())
    print("\nSummary statistics:\n", data.describe())

def preprocess_data(data):
    """Preprocess the dataset."""
    # Fill missing values in TotalCharges
    data['TotalCharges'] = data['TotalCharges'].replace(' ', '0').astype(float)
    # Convert categorical variables to numerical using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)
    return data

def split_data(data):
    """Split the dataset into features (X) and target variable (y)."""
    X = data.drop('Churn_Yes', axis=1)
    y = data['Churn_Yes']
    return X, y

def train_model(X_train, X_test, y_train, y_test):
    """Train a Random Forest Classifier and evaluate it."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = 'customer_data.csv'

    # Load the data
    data = load_data(file_path)

    # Explore the data
    explore_data(data)

    # Preprocess the data
    data = preprocess_data(data)

    # Split data into training and testing sets
    X, y = split_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train and evaluate the model
    train_model(X_train, X_test, y_train, y_test)

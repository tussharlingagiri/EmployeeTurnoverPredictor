import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load data
data = pd.read_csv("modified_employee_turnover.csv")

# Check for missing values
if data.isnull().sum().sum() > 0:
    print("Error: Missing values detected in the dataset. Please clean the data and try again.")
    exit()

# Display class distribution
print("Class Distribution:\n", data['Employee_Turnover'].value_counts(normalize=True))

# Split features and target
X = data.drop('Employee_Turnover', axis=1)
y = data['Employee_Turnover']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Fixed: Use transform instead of fit_transform

# Define hyperparameter grid
C_values = np.logspace(-3, 1, 20)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Hyperparameter tuning
best_f1 = 0
best_params = {}

for C in C_values:
    for l1_ratio in l1_ratios:
        f1_scores = []
        for train_idx, val_idx in kf.split(X_train_scaled):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = LogisticRegression(
                penalty='elasticnet',
                C=C,
                l1_ratio=l1_ratio,
                solver='saga',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            f1_scores.append(f1_score(y_val, y_pred))

        mean_f1 = np.mean(f1_scores)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_params = {'C': C, 'l1_ratio': l1_ratio}

# Print best parameters
print(f"Best Parameters: C={best_params['C']:.4f}, l1_ratio={best_params['l1_ratio']}")
print(f"Best Cross-Validated F1-Score: {best_f1:.4f}")

# Train final model
final_model = LogisticRegression(
    penalty='elasticnet',
    C=best_params['C'],
    l1_ratio=best_params['l1_ratio'],
    solver='saga',
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
final_model.fit(X_train_scaled, y_train)

# Evaluate on training set
y_train_pred = final_model.predict(X_train_scaled)
print("\nTraining Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"F1-Score: {f1_score(y_train, y_train_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))

# Evaluate on test set
y_test_pred = final_model.predict(X_test_scaled)
print("\nTest Set Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# Create results DataFrame for test set
test_results = X_test.copy()
test_results['Actual_Turnover'] = y_test
test_results['Predicted_Turnover'] = y_test_pred
test_results['Turnover_Probability'] = final_model.predict_proba(X_test_scaled)[:, 1]
test_results['Index'] = X_test.index

# Display employees predicted to leave
leavers = test_results[test_results['Predicted_Turnover'] == 1]
if not leavers.empty:
    columns_to_show = [
        'Index', 'Actual_Turnover', 'Predicted_Turnover', 'Turnover_Probability',
        'Job_Satisfaction', 'Years_At_Company', 'Monthly_Income', 'Department'
    ]
    # Ensure columns exist in the dataset
    columns_to_show = [col for col in columns_to_show if col in leavers.columns]
    print("\nEmployees Predicted to Leave:")
    print(leavers[columns_to_show].sort_values(by='Turnover_Probability', ascending=False))
    print(f"\nTotal number of employees predicted to leave: {len(leavers)}")
else:
    print("\nNo employees are predicted to leave.")
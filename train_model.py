from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

df = pd.read_csv("new_data.csv")

X = df.drop(columns=["Depression_Category", "Anxiety_Category", "Stress_Category"])

y_depression = df["Depression_Category"]
y_anxiety = df["Anxiety_Category"]
y_stress = df["Stress_Category"]

X_train, X_test, y_train_depression, y_test_depression = train_test_split(
    X, y_depression, test_size=0.2, random_state=42
)
_, _, y_train_anxiety, y_test_anxiety = train_test_split(
    X, y_anxiety, test_size=0.2, random_state=42
)
_, _, y_train_stress, y_test_stress = train_test_split(
    X, y_stress, test_size=0.2, random_state=42
)


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train_depression shape:", y_train_depression.shape)
print("y_test_depression shape:", y_test_depression.shape)


# Initialize the Random Forest classifier
rf_depression = RandomForestClassifier(random_state=42)
rf_anxiety = RandomForestClassifier(random_state=42)
rf_stress = RandomForestClassifier(random_state=42)

# Train the models on the respective target variables
rf_depression.fit(X_train, y_train_depression)
rf_anxiety.fit(X_train, y_train_anxiety)
rf_stress.fit(X_train, y_train_stress)

# Predict the test set for each target
y_pred_depression = rf_depression.predict(X_test)
y_pred_anxiety = rf_anxiety.predict(X_test)
y_pred_stress = rf_stress.predict(X_test)

# Evaluate the models using accuracy
accuracy_depression = accuracy_score(y_test_depression, y_pred_depression)
accuracy_anxiety = accuracy_score(y_test_anxiety, y_pred_anxiety)
accuracy_stress = accuracy_score(y_test_stress, y_pred_stress)

# Print the accuracies for each model
print(f"Depression Model Accuracy: {accuracy_depression * 100:.2f}%")
print(f"Anxiety Model Accuracy: {accuracy_anxiety * 100:.2f}%")
print(f"Stress Model Accuracy: {accuracy_stress * 100:.2f}%")


import pickle

# Save the depression model
with open("models/depression_model.pkl", "wb") as file:
    pickle.dump(rf_depression, file)

# Save the anxiety model
with open("models/anxiety_model.pkl", "wb") as file:
    pickle.dump(rf_anxiety, file)

# Save the stress model
with open("models/stress_model.pkl", "wb") as file:
    pickle.dump(rf_stress, file)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Plot confusion matrices for each model
plot_confusion_matrix(y_test_depression, y_pred_depression, "Depression")
plot_confusion_matrix(y_test_anxiety, y_pred_anxiety, "Anxiety")
plot_confusion_matrix(y_test_stress, y_pred_stress, "Stress")

# Classification report for each model
print("Depression Model Classification Report")
print(classification_report(y_test_depression, y_pred_depression))

print("Anxiety Model Classification Report")
print(classification_report(y_test_anxiety, y_pred_anxiety))

print("Stress Model Classification Report")
print(classification_report(y_test_stress, y_pred_stress))


# Bar chart to visualize precision, recall, and F1-score from the classification report
def plot_classification_report(y_true, y_pred, title):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = (
        pd.DataFrame(report).transpose().iloc[:-1, :3]
    )  # Removing the 'support' and 'accuracy' row
    df_report[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(10, 6))
    plt.title(f"Precision, Recall, F1-Score for {title} Model")
    plt.ylabel("Score")
    plt.show()


plot_classification_report(y_test_depression, y_pred_depression, "Depression")
plot_classification_report(y_test_anxiety, y_pred_anxiety, "Anxiety")
plot_classification_report(y_test_stress, y_pred_stress, "Stress")


# Feature Importance plot for each model
def plot_feature_importance(model, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {title} Model")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
    plt.show()


# Plot feature importances
plot_feature_importance(rf_depression, "Depression")
plot_feature_importance(rf_anxiety, "Anxiety")
plot_feature_importance(rf_stress, "Stress")

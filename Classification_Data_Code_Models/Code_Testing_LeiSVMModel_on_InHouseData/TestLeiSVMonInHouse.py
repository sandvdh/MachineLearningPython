import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load new data
new_data = pd.read_csv('preprocessed_InHouseData.csv')  # Replace with your new data file

# Remove PVDF and replace LDPE and HDPE with PE in the dataset
new_data = new_data[new_data['Plastic'] != 'PVDF']
new_data['Plastic'] = new_data['Plastic'].replace({'LDPE': 'PE', 'HDPE': 'PE'})

# Prepare features and labels
X = new_data.drop(columns=["Plastic"])  # Features (wavenumber columns)
y = new_data["Plastic"]  # Labels (plastic types)

# Apply the same scaling
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)

# Save predictions to a CSV file
predictions_df = pd.DataFrame({'True Label': y, 'Predicted Label': y_pred})
predictions_df.to_csv('new_data_predictions_allsamples.csv', index=False)
print("Predictions saved as 'new_data_predictions_allsamples.csv'.")

# Evaluate the model
print("\nAccuracy Score:")
print(accuracy_score(y, y_pred))

# Generate the classification report
print("\nClassification Report:")
report = classification_report(y, y_pred, zero_division=0, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Round the numbers to two decimal places
report_df = report_df.round(2)

# Save the classification report as a CSV file
report_df.to_csv("classification_report_allsamples.csv")
print(report_df)

# Identify classes with NaN or zero metrics
print("\nClasses with NaN or zero precision/recall/f1-score:")
for label, metrics in report.items():
    if isinstance(metrics, dict):  # Skips 'accuracy' entry in the report
        precision, recall, f1_score = metrics['precision'], metrics['recall'], metrics['f1-score']

        if precision == 0.0 or recall == 0.0 or f1_score == 0.0:
            print(
                f"Label: {label} has undefined metrics (Precision: {precision}, Recall: {recall}, F1-score: {f1_score}).")

# Confusion Matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y, y_pred, labels=model.classes_)
print(conf_matrix)

# Calculate and save misclassified samples
misclassified = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
misclassified_df = pd.DataFrame({'Plastic': model.classes_, 'Misclassified': misclassified})
misclassified_df.to_csv("misclassified_samples_allsamples.csv", index=False)
print(misclassified_df)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(ax=ax, cmap=plt.cm.Blues)

# Add red border around off-diagonal elements greater than zero
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        if i != j and conf_matrix[i, j] > 0:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))

plt.title('In-house data tested on Lei et al. SVM model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

print("End of script.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load data
data = pd.read_csv('2022-03-31-20aug-1cm-720-1800-20baseline.csv')
lei_df = data.replace(
    ['acrylonitrile butadiene styrene', 'poly (methyl methacrylate)', 'polycarbonate', 'polyester', 'polyethylene', 'polyethylene terephthalate', 'polyoxymethylene', 'polypropylene', 'polystyrene', 'polytetrafluoroethylene', 'polyurethane', 'polyvinyl chloride'],
    ['ABS', 'PMMA', 'PC', 'PES', 'PE', 'PET', 'POM', 'PP', 'PS', 'PTFE', 'PU', 'PVC'])

# Extract features (spectra) and labels
X = lei_df.iloc[:, 3:]  # Features (spectra)
y = lei_df['plastic']   # Labels (plastic types)

# Stratified sampling to keep class distribution similar in both training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Standardize the features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=3, leaf_size=5)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'Lei_KNN_model.pkl')
joblib.dump(scaler, 'Lei_Robust_scaler.pkl')
print("Model and scaler saved as 'Lei_KNN_model.pkl' and 'Lei_KNN_scaler.pkl'.")

# Predict on the test set
y_pred_test = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_test)
print("Model Accuracy:", accuracy * 100, "%")

# Generate the classification report with two decimal places
report_dict = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)
classification_report_df = pd.DataFrame(report_dict).transpose()

# Format the DataFrame to display two decimal places using DataFrame.map
classification_report_df = classification_report_df.apply(lambda col: col.map(lambda x: f'{x:.2f}' if isinstance(x, (float, int)) else x))

# Save the classification report to a CSV file
classification_report_df.to_csv('new_classification_report_Lei_KNN.csv', index=True)
print("Classification report saved as 'new_classification_report_Lei_KNN.csv'.")

# Create DataFrame with true and predicted labels
results_df = pd.DataFrame({'True Label': y_test.values, 'Predicted Label': y_pred_test})

# Identify classes with no predicted samples
classes_with_no_predictions = np.setdiff1d(y_test.unique(), y_pred_test)
print("Classes with no predictions:", classes_with_no_predictions)

# Filter the DataFrame for NaN-inducing samples
nan_samples = results_df[results_df['True Label'].isin(classes_with_no_predictions)]
results_df['NaN Inducing'] = results_df['True Label'].isin(classes_with_no_predictions)
results_df['Misclassified'] = results_df['True Label'] != results_df['Predicted Label']

# Separate NaN-inducing and other misclassified samples
nan_inducing_samples = results_df[results_df['NaN Inducing']]
misclassified_samples = results_df[results_df['Misclassified'] & ~results_df['NaN Inducing']]

# Save misclassified samples as CSV
nan_inducing_samples.to_csv('nan_inducing_samples_KNN.csv', index=False)
misclassified_samples.to_csv('misclassified_samples_KNN.csv', index=False)
print("NaN-inducing samples saved as 'nan_inducing_samples_KNN.csv'.")
print("Other misclassified samples saved as 'misclassified_samples_KNN.csv'.")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y))
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)

# Add red border around off-diagonal elements greater than zero
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        if i != j and conf_matrix[i, j] > 0:
            disp.ax_.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))

plt.title('Confusion Matrix optimized KNN')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Misclassification Summary
misclassification_summary = []

# Iterate through the confusion matrix to find misclassifications
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        if i != j and conf_matrix[i, j] > 0:
            misclassification_summary.append({
                'True Label': model.classes_[i],
                'Predicted Label': model.classes_[j],
                'Count': conf_matrix[i, j]
            })

# Convert the list to a DataFrame
misclassification_summary_df = pd.DataFrame(misclassification_summary)

# Save the summary as a CSV file
misclassification_summary_df.to_csv('misclassification_summary_KNN.csv', index=False)

print("Misclassification summary saved as 'misclassification_summary_KNN.csv'.")
print("End of script")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_csv('2022-03-31-20aug-1cm-720-1800-20baseline.csv')
lei_df = data.replace(['acrylonitrile butadiene styrene', 'poly (methyl methacrylate)', 'polycarbonate', 'polyester', 'polyethylene', 'polyethylene terephthalate', 'polyoxymethylene', 'polypropylene', 'polystyrene', 'polytetrafluoroethylene', 'polyurethane', 'polyvinyl chloride'],
                      ['ABS', 'PMMA', 'PC', 'PES', 'PE', 'PET', 'POM', 'PP', 'PS', 'PTFE', 'PU', 'PVC'])

X = lei_df.drop(["plastic", "supplier", "sampleid"], axis=1)
y = lei_df["plastic"]

# Stratified sampling to keep class distribution similar in both training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Standardize the features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=346, max_depth=15, random_state=0)
model.fit(X_train, y_train)

# Predict on the test set
y_pred_test = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_test)
print("Model Accuracy:", accuracy * 100, "%")

# Classification Report with 2 decimal points
report_dict = classification_report(y_test, y_pred_test, zero_division=0, output_dict=True)

# Convert the report to DataFrame and round to 2 decimal places
new_classification_report_Lei_RF = pd.DataFrame(report_dict).transpose()
new_classification_report_Lei_RF = new_classification_report_Lei_RF.round(2)

# Save the classification report as a CSV file
new_classification_report_Lei_RF.to_csv('new_classification_report_Lei_RF.csv', index=True)
print("Classification report saved as 'new_classification_report_Lei_RF.csv'.")

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
nan_inducing_samples.to_csv('nan_inducing_samples.csv', index=False)
misclassified_samples.to_csv('misclassified_samples.csv', index=False)
print("NaN-inducing samples saved as 'nan_inducing_samples.csv'.")
print("Other misclassified samples saved as 'misclassified_samples.csv'.")

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

plt.title('Confusion Matrix optimized RF')
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
misclassification_summary_df.to_csv('misclassification_summary.csv', index=False)

print("Misclassification summary saved as 'misclassification_summary.csv'.")

print("End of script")


# Feature selection using SelectKBest with mutual information
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load the cleaned data
data_final = pd.read_csv('preprocessed_InHouseData.csv')

# Inspect the shape of the data
print("Shape of the cleaned data:", data_final.shape)

# Prepare features and labels
X = data_final.drop(columns=["Plastic"])  # Features (wavenumber columns)
y = data_final["Plastic"]  # Labels (plastic types)

# Check the distribution of the classes in the target variable
class_counts = y.value_counts()
print("Class distribution:")
print(class_counts)

# Identify classes with only one sample
rare_classes = class_counts[class_counts == 1]
if not rare_classes.empty:
    print("\nClasses with only one sample:")
    print(rare_classes)

    # Optionally, you can remove these rare classes from the dataset
    data_final = data_final[~data_final['Plastic'].isin(rare_classes.index)]

    # Update features and labels after removing rare classes
    X = data_final.drop(columns=["Plastic"])  # Features
    y = data_final["Plastic"]  # Labels


# Inspect the shapes of X and y
print("\nShape of features (X):", X.shape)
print("Shape of labels (y):", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

k_best = SelectKBest(score_func=mutual_info_classif, k=130)  # Adjust 'k' to select the desired number of features
X_train = k_best.fit_transform(X_train, y_train)
X_test = k_best.transform(X_test)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Generate the classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

# Convert the classification report to a DataFrame
report_df = pd.DataFrame(report).transpose()

# Round the numbers to two decimal places
report_df = report_df.round(2)

# Save the classification report as a CSV file
report_df.to_csv("classification_report.csv")
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
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(conf_matrix)

# Calculate and save misclassified samples
misclassified = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
misclassified_df = pd.DataFrame({'Plastic': model.classes_, 'Misclassified': misclassified})
misclassified_df.to_csv("misclassified_samples.csv", index=False)
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

plt.title('Confusion Matrix optimized RF')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()



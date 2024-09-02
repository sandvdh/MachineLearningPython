import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the cleaned data
data_final = pd.read_csv('preprocessed_InHouseData.csv')

# Prepare features and labels
X = data_final.drop(columns=["Plastic"])  # Features (wavenumber columns)
y = data_final["Plastic"]  # Labels (plastic types)

# Check the distribution of the classes in the target variable
class_counts = y.value_counts()
print("Class distribution:")
print(class_counts)

# Identify and remove classes with 3 or fewer samples
rare_classes = class_counts[class_counts <= 3]
if not rare_classes.empty:
    print("\nClasses with three or fewer samples:")
    print(rare_classes)

    # Remove rare classes from the dataset
    data_final = data_final[~data_final['Plastic'].isin(rare_classes.index)]

    # Update features and labels after removing rare classes
    X = data_final.drop(columns=["Plastic"])  # Features
    y = data_final["Plastic"]  # Labels

# Check the updated class distribution
class_counts = y.value_counts()
print("\nUpdated Class Distribution:")
print(class_counts)

# Inspect the shapes of X and y
print("\nShape of features (X):", X.shape)
print("Shape of labels (y):", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up Stratified K-Fold Cross-Validation with 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=skf,
                           n_jobs=-1,
                           verbose=2,
                           scoring='accuracy')

# Fit the GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Retrieve the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train the final model with the best parameters
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Make predictions with the optimized model
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Optimized Model Accuracy:", accuracy * 100, "%")

# Generate and save the classification report
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df.round(2)
report_df.to_csv("new_classification_report_InHouse_RF_3orless.csv", index=True)
print(report_df)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
print("\nConfusion Matrix:")
print(conf_matrix)

# Misclassification Summary
misclassified = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
misclassified_df = pd.DataFrame({'Plastic': best_model.classes_, 'Misclassified': misclassified})
misclassified_df.to_csv("misclassified_samples_InHouse_RF_3orless.csv", index=False)
print(misclassified_df)

# Display Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_model.classes_)
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

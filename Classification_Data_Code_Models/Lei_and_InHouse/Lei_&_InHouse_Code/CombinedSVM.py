import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import RobustScaler, MinMaxScaler

# Load the cleaned data with the correct delimiter
data_final = pd.read_csv('Combined_Lei_InHouse_data.csv', delimiter=';', decimal=',')

# Inspect the DataFrame
print("DataFrame columns:", data_final.columns)
print("DataFrame head:\n", data_final.head())

# Prepare features and labels
if 'Plastic' in data_final.columns:
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

    # Apply SelectKBest with mutual_info_classif to select the top k features
    selector = SelectKBest(score_func=mutual_info_classif, k=500)
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
        'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
        'kernel': ['rbf', 'linear']  # Kernel type, RBF is commonly used for non-linear classification
    }

    # Initialize StratifiedKFold
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize GridSearchCV with stratified k-fold
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=stratified_kfold, scoring='accuracy', verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Output the best parameters found by GridSearchCV
    print("\nBest Parameters found by GridSearchCV:")
    print(grid_search.best_params_)

    # Train the best estimator found by GridSearchCV
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Make predictions
    y_pred = best_model.predict(X_test)

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
    report_df.to_csv("classification_report_combined_SVMOpt.csv")
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
    conf_matrix = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    print(conf_matrix)

    # Calculate and save misclassified samples
    misclassified = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
    misclassified_df = pd.DataFrame({'Plastic': best_model.classes_, 'Misclassified': misclassified})
    misclassified_df.to_csv("misclassified_samples_combined_SVMOpt.csv", index=False)
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

    plt.title('Combined Lei and InHouse dataset - SVC')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Plastic' not found in the DataFrame.")

print('End of script.')

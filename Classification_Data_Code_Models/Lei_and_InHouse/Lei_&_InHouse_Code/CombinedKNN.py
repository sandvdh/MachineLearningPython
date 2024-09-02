import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import joblib

# Load the cleaned data with the correct delimiter
data_final = pd.read_csv('Combined_Lei_InHouse_data.csv', delimiter= ';', decimal=',')

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize the scaler
    scaler = RobustScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler to a file
    joblib.dump(scaler, 'CombinedData_Robust_scaler.pkl')

    # Define the KNN model
    knn = KNeighborsClassifier()

    # Define the hyperparameters and grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Perform GridSearchCV with stratified cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from GridSearch
    best_knn = grid_search.best_estimator_

    # Save the best model to a file
    joblib.dump(best_knn, 'Final_CombinedLeiInHouse_KNN_model.pkl')

    # Evaluate the best model on the test set
    y_pred = best_knn.predict(X_test)
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=best_knn.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues)

    # Add red border around off-diagonal elements greater than zero
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if i != j and conf_matrix[i, j] > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none'))

    plt.title('Confusion Matrix for Best KNN Model')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()





import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# Loadingthe CSV files
df_w1 = pd.read_csv('w1.csv')  # Load the first dataset
df_w2 = pd.read_csv('w2.csv')  # Load the second dataset
df_w3 = pd.read_csv('w3.csv')  # Load the third dataset
df_w4 = pd.read_csv('w4.csv')  # Load the fourth dataset
# Combine the datasets into a single DataFrame
combineddataset = pd.concat([df_w1, df_w2, df_w3, df_w4], ignore_index=True)

# Save the combined DataFrame into a new CSV file
combineddataset.to_csv('combined_data.csv', index=False)
# Shuffle the data
shuffled_df = combineddataset.sample(n=len(combineddataset)).reset_index(drop=True)

# Save the shuffled DataFrame to a new CSV file
shuffled_df.to_csv('all_data.csv', index=False)

# Separate features (X) and the target variable (y)
X = shuffled_df.iloc[:, :-1]  # Select all columns except the last one as features
y = shuffled_df.iloc[:, -1]   # Select the last column as the target variable
# Split the data into 70% train and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Iitialize the SVM model
clf = svm.SVC()
# Train the model on the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Measure the accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model on the test set
print(f"The model accuracy for the 70/30 split test is: {accuracy * 100:.2f}%")

# Initialize the SVM model for cross-validation
clf = svm.SVC()

# Perform 10-fold cross-validation on the entire dtaset
scores = cross_val_score(clf, X, y, cv=10)

# Output the cross-validation accuracy scores
formatted_scores = [f"{score * 100:.2f}%" for score in scores]
print(f"Cross-validation accuracy scores: {formatted_scores}")
print(f"Mean cross-validation accuracy: {scores.mean() * 100:.2f}%")

# Initialize the SVM model with RBF kernel for hyperparameter tuning
clf = svm.SVC(kernel='rbf')

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],        # Range of values for the regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Range of values for the gamma parameter
    'kernel': ['rbf']               # Use the RBF kernel
}
# Initialize GridSearchCV to perform hyperparameter tuning with 10-fold cross-validation
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, cv=10)

# Fit GridSearchCV to the training data
grid.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
print(f"Best parameters found: {grid.best_params_}")

# Use the best estimator to make predictions on the test set
best_clf = grid.best_estimator_

# Predict on the test set using the best model fromGridSarchCV
y_pred = best_clf.predict(X_test)

# Measure the accuracy of the best model on the test set
best_accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy with the best hyperparameters: {best_accuracy * 100:.2f}%")

# Use the best estimator found by GridSearchCV to reinitialize the SVM model
clf = svm.SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel='rbf')
# Train the reinitialized model on the training data
clf.fit(X_train, y_train)
# Make predictions on the test set using the reinitialized model
y_pred = clf.predict(X_test)
# Measure the accuracy of the reinitialized model on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Updated model accuracy with optimal hyperparameters (train-test split): {accuracy * 100:.2f}%")
# Perform 10-fold cross-validation on the entire dataset using the reinitialized model
scores = cross_val_score(clf, X, y, cv=10)
#Output the cross-validation accuracy scores
formatted_scores = [f"{score * 100:.2f}%" for score in scores]
print(f"Updated cross-validation accuracy scores: {formatted_scores}")
print(f"Updated mean cross-validation accuracy: {scores.mean() * 100:.2f}%")
# Select the 100 best features
selector = SelectKBest(f_classif, k=100)
# Fit the selector to the data and transform X
X_new = selector.fit_transform(X, y)
# Split the reduced feature set into 70% train and 30% test sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.3, random_state=1)
# Train the model with selected features and optimal hyperparameters
clf = svm.SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel='rbf')
clf.fit(X_train_new, y_train_new)
# Make predictions on the test set
y_pred_new = clf.predict(X_test_new)

# Measure the accuracy
accuracy_new = accuracy_score(y_test_new, y_pred_new)
print(f"Model accuracy with feature selection (train-test split): {accuracy_new * 100:.2f}%")

# Perform 10-fold cross-validation on the reduced feature set
scores_new = cross_val_score(clf, X_new, y, cv=10)

# Output the cross-validation accuracy scores
formatted_scores_new = [f"{score * 100:.2f}%" for score in scores_new]
print(f"Cross-validation accuracy scores with feature selection: {formatted_scores_new}")
print(f"Mean cross-validation accuracy with feature selection: {scores_new.mean() * 100:.2f}%")

# Apply PCA to reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split the PCA-reduced data into 70% train and 30% test sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=1)

# Train the model with PCA-reduced features and optimal hyperparameters
clf_pca = svm.SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel='rbf')
clf_pca.fit(X_train_pca, y_train_pca)

# Make predictions on the test set
y_pred_pca = clf_pca.predict(X_test_pca)
# Measure the accuracy
accuracy_pca = accuracy_score(y_test_pca, y_pred_pca)
print(f"Model accuracy with PCA (train-test split): {accuracy_pca * 100:.2f}%")
# Perform 10-fold cross-validation on the PCA-reduced data
scores_pca = cross_val_score(clf_pca, X_pca, y, cv=10)

# Output the cross-validation accuracy scores
formatted_scores_pca = [f"{score * 100:.2f}%" for score in scores_pca]
print(f"Cross-validation accuray scores with PCA: {formatted_scores_pca}")
print(f"Mean cross-validtion accuracy with PCA: {scores_pca.mean() * 100:.2f}%")

# Initialize the SGDClassifier
sgd_clf = SGDClassifier(random_state=1)

# Train the SGD model using the train-test split
sgd_clf.fit(X_train, y_train)
y_pred_sgd = sgd_clf.predict(X_test)
sgd_train_test_acc = accuracy_score(y_test, y_pred_sgd)
print(f"SGDClassiier accuracy with train-test split: {sgd_train_test_acc * 100:.2f}%")

# Perform 10-fold cross-validation
sgd_cv_scores = cross_val_score(sgd_clf, X, y, cv=10)
sgd_cv_acc = sgd_cv_scores.mean() * 100
print(f"SGDClassifier cross-validatio accuracy: {sgd_cv_acc:.2f}%")

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=1)

# Train the RandomForest model using the train-test split
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_train_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"RandomForestClassifier ccuracy with train-test split: {rf_train_test_acc * 100:.2f}%")

# Perform 10-fold cross-validation
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=10)
rf_cv_acc = rf_cv_scores.mean() * 100
print(f"RandomForestClassifier cros-validation accuracy: {rf_cv_acc:.2f}%")

# Initialize the MLPClassifier
mlp_clf = MLPClassifier(random_state=1, max_iter=300)

# Train the MLP model using the train-test split
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)
mlp_train_test_acc = accuracy_score(y_test, y_pred_mlp)
print(f"MLPClassifier accuracy with train-test split: {mlp_train_test_acc * 100:.2f}%")
# Perform 10-fold cross-validation
mlp_cv_scores = cross_val_score(mlp_clf, X, y, cv=10)
mlp_cv_acc = mlp_cv_scores.mean() * 100
print(f"MLPClassifier cross-valdation accuracy: {mlp_cv_acc:.2f}%")

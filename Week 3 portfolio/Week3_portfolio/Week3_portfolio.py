import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold

# Load the datasets (replace 'boning.csv' and 'slicing.csv' with the actual file names)
df_boning = pd.read_csv('Boning.csv')
df_slicing = pd.read_csv('Slicing.csv')
# Define the columns to extract
cols_to_extract = ['Frame', 'Right Lower Leg x', 'Right Lower Leg y', 'Right Lower Leg z',
                   'Left Lower Leg x', 'Left Lower Leg y', 'Left Lower Leg z']

# Extract these columns
df_boning = df_boning[cols_to_extract]
df_slicing = df_slicing[cols_to_extract]
# Add the class labels
df_boning['class'] = 0
df_slicing['class'] = 1

# Combine the datasets
df_combined = pd.concat([df_boning, df_slicing], ignore_index=True)

# Preview the combined dataset
print(df_combined.head())
df_combined.to_csv('combined_data.csv', index=False)

# Function to compute RMS


def compute_rms(df, col1, col2):
    return np.sqrt(df[col1]**2 + df[col2]**2)


# Compute RMS columns for Right Lower Leg
df_combined['rms_xy_right_lower_leg'] = compute_rms(df_combined, 'Right Lower Leg x', 'Right Lower Leg y')
df_combined['rms_yz_right_lower_leg'] = compute_rms(df_combined, 'Right Lower Leg y', 'Right Lower Leg z')
df_combined['rms_zx_right_lower_leg'] = compute_rms(df_combined, 'Right Lower Leg z', 'Right Lower Leg x')
df_combined['rms_xyz_right_lower_leg'] = np.sqrt(df_combined['Right Lower Leg x']**2 +
                                                 df_combined['Right Lower Leg y']**2 +
                                                 df_combined['Right Lower Leg z']**2)

# Compute RMS columns for Left Lower Leg
df_combined['rms_xy_left_lower_leg'] = compute_rms(df_combined, 'Left Lower Leg x', 'Left Lower Leg y')
df_combined['rms_yz_left_lower_leg'] = compute_rms(df_combined, 'Left Lower Leg y', 'Left Lower Leg z')
df_combined['rms_zx_left_lower_leg'] = compute_rms(df_combined, 'Left Lower Leg z', 'Left Lower Leg x')
df_combined['rms_xyz_left_lower_leg'] = np.sqrt(df_combined['Left Lower Leg x']**2 +
                                                df_combined['Left Lower Leg y']**2 +
                                                df_combined['Left Lower Leg z']**2)


# Function to compute Roll and Pitch
def compute_roll_pitch(df, accelX, accelY, accelZ):
    roll = 180 * np.arctan2(df[accelY], np.sqrt(df[accelX]**2 + df[accelZ]**2)) / np.pi
    pitch = 180 * np.arctan2(df[accelX], np.sqrt(df[accelY]**2 + df[accelZ]**2)) / np.pi
    return roll, pitch


# Compute Roll and Pitch for Right Lower Leg
df_combined['roll_right_lower_leg'], df_combined['pitch_right_lower_leg'] = compute_roll_pitch(df_combined,
                                                                                                'Right Lower Leg x',
                                                                                                'Right Lower Leg y',
                                                                                                'Right Lower Leg z')

# Compute Roll and Pitch for Left Lower Leg
df_combined['roll_left_lower_leg'], df_combined['pitch_left_lower_leg'] = compute_roll_pitch(df_combined,
                                                                                             'Left Lower Leg x',
                                                                                             'Left Lower Leg y',
                                                                                             'Left Lower Leg z')

# Preview the updated dataset
print(df_combined.head())
df_combined.to_csv('combined_data_composite.csv', index=False)


# Function to compute the statistical features using pandas and numpy
def compute_features(data):
    features = {}
    features['mean'] = data.mean()
    features['std'] = data.std()
    features['min'] = data.min()
    features['max'] = data.max()

    # Area Under the Curve (AUC) - Sum of the values as a simple approximation
    features['auc'] = data.sum()

    # Peaks - A basic approach, counting how many times a value is greater than its neighbors
    features['peaks'] = ((data.shift(1) < data) & (data.shift(-1) < data)).sum()

    return features


# List of dictionaries to hold the computed features for each segment
features_list = []

# List of columns to compute features for (exclude 'Frame' and 'Class')
columns_to_compute = df_combined.columns[1:-1]  # Columns 2-19

# Process each segment of 60 frames (1 minute)
for i in range(0, len(df_combined), 60):
    segment = df_combined.iloc[i:i + 60]
    segment_features = {}

    # Add frame label to the segment features
    segment_features['Frame'] = df_combined['Frame'].iloc[i]

    for col in columns_to_compute:
        features = compute_features(segment[col])
        for key, value in features.items():
            segment_features[f'{col}_{key}'] = value

    # Add class label to the segment features
    segment_features['class'] = df_combined['class'].iloc[i]

    # Append the features dictionary to the list
    features_list.append(segment_features)

# Convert the list of dictionaries to a DataFrame
feature_df = pd.DataFrame(features_list)

# Save the features
feature_df.to_csv('feature_data.csv', index=False)

# Drop the 'Frame' column and separate features (X) and target variable (y)
X = feature_df.drop(columns=['Frame', 'class'])
y = feature_df['class']

# Preview the feature dataframe
print(feature_df.head())


# Split the data into 70% train and 30% test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize the SVM model
clf = svm.SVC(kernel='rbf')

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Measure the accuracy
accuracy_70_30 = accuracy_score(y_test, y_pred)
print(f"SVM Model accuracy with 70/30 train-test split: {accuracy_70_30 * 100:.2f}%")


# Perform 10-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=10)

# Output the cross-validation accuracy scores
cv_mean_accuracy = cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy: {cv_mean_accuracy:.2f}%")


# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Initialize GridSearchCV
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2, cv=10)

# Fit GridSearchCV to the training data
grid.fit(X_train, y_train)

# Print the best parameters found by GridSearchCV
best_params = grid.best_params_
print(f"Best parameters found: {best_params}")

# Use the best estimator to make predictions on the test set
best_clf = grid.best_estimator_
y_pred_best = best_clf.predict(X_test)

# Measure the accuracy with the best model on the test split
best_accuracy_70_30 = accuracy_score(y_test, y_pred_best)
print(f"SVM Model accuracy with optimal hyperparameters (70/30 split): {best_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with the best model
best_cv_scores = cross_val_score(best_clf, X, y, cv=10)
best_cv_mean_accuracy = best_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with optimal hyperparameters: {best_cv_mean_accuracy:.2f}%")


# Initialize VarianceThreshold to remove constant features (threshold=0 removes only constant features)
constant_filter = VarianceThreshold(threshold=0)
X_filtered = constant_filter.fit_transform(X)

# Check how many features were removed
removed_features = X.shape[1] - X_filtered.shape[1]
print(f"Number of constant features removed: {removed_features}")

# Now, use X_filtered instead of X for further processing


# Select the 10 best features
selector = SelectKBest(f_classif, k=10)
X_best = selector.fit_transform(X, y)

# Split the data again with the selected features
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(X_best, y, test_size=0.3, random_state=1)

# Train the model with the selected features and optimal hyperparameters
best_clf.fit(X_train_best, y_train_best)
y_pred_best = best_clf.predict(X_test_best)

# Measure the accuracy with the selected features on the test split
best_feature_accuracy_70_30 = accuracy_score(y_test_best, y_pred_best)
print(f"SVM Model accuracy with selected features (70/30 split): {best_feature_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with the selected features
best_feature_cv_scores = cross_val_score(best_clf, X_best, y, cv=10)
best_feature_cv_mean_accuracy = best_feature_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with selected features: {best_feature_cv_mean_accuracy:.2f}%")


# Apply PCA to reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split the PCA-reduced data
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=1)

# Train the model with PCA-reduced features and optimal hyperparameters
best_clf.fit(X_train_pca, y_train_pca)
y_pred_pca = best_clf.predict(X_test_pca)

# Measure the accuracy with PCA on the test split
best_pca_accuracy_70_30 = accuracy_score(y_test_pca, y_pred_pca)
print(f"SVM Model accuracy with PCA (70/30 split): {best_pca_accuracy_70_30 * 100:.2f}%")

# Perform 10-fold cross-validation with PCA
best_pca_cv_scores = cross_val_score(best_clf, X_pca, y, cv=10)
best_pca_cv_mean_accuracy = best_pca_cv_scores.mean() * 100
print(f"SVM Mean cross-validation accuracy with PCA: {best_pca_cv_mean_accuracy:.2f}%")

# Initialize the SGDClassifier
sgd_clf = SGDClassifier(random_state=1)
sgd_clf.fit(X_train, y_train)
y_pred_sgd = sgd_clf.predict(X_test)
sgd_train_test_acc = accuracy_score(y_test, y_pred_sgd)
print(f"SGDClassifier accuracy with train-test split: {sgd_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
sgd_cv_scores = cross_val_score(sgd_clf, X, y, cv=10)
sgd_cv_acc = sgd_cv_scores.mean() * 100
print(f"SGDClassifier cross-validation accuracy: {sgd_cv_acc:.2f}%")

# Initialize the RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=1)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
rf_train_test_acc = accuracy_score(y_test, y_pred_rf)
print(f"RandomForestClassifier accuracy with train-test split: {rf_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
rf_cv_scores = cross_val_score(rf_clf, X, y, cv=10)
rf_cv_acc = rf_cv_scores.mean() * 100
print(f"RandomForestClassifier cross-validation accuracy: {rf_cv_acc:.2f}%")

# Initialize the MLPClassifier
mlp_clf = MLPClassifier(random_state=1, max_iter=500)
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)
mlp_train_test_acc = accuracy_score(y_test, y_pred_mlp)
print(f"MLPClassifier accuracy with train-test split: {mlp_train_test_acc * 100:.2f}%")

# 10-fold cross-validation accuracy
mlp_cv_scores = cross_val_score(mlp_clf, X, y, cv=10)
mlp_cv_acc = mlp_cv_scores.mean() * 100
print(f"MLPClassifier cross-validation accuracy: {mlp_cv_acc:.2f}%")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the CSV dataset into a pandas DataFrame
file_path = 'CCPP.csv'
data = pd.read_csv(file_path)

# Define a function to categorize the 'PE' values
def categorize_pe(value):
    if value < 430:
        return 1
    elif 20 <= value < 450:
        return 2
    elif 30 <= value < 470:
        return 3
    elif 40 <= value < 490:
        return 4
    else:
        return 5

# Step 3: Apply the function to the 'PE' column
data['PE_Label'] = data['PE'].apply(categorize_pe)

# Step 4: Write the converted DataFrame to a new CSV file
output_path = 'converted_CCPP.csv'  # This will save in the same directory as your script
data.to_csv(output_path, index=False)

# Optional: Print the first few rows to verify the result
print(data.head())

# Load the CSV dataset (if not already loaded)
file_path = 'converted_CCPP.csv'  # Use the path where you saved your file
data = pd.read_csv(file_path)

# Plot the distribution of the classes in a bar chart
class_distribution = data['PE_Label'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='blue')
plt.title('Distribution of Energy Output Classes (PE_Label)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)  # Keep the x labels horizontal
plt.show()

# Load the dataset
data = pd.read_csv('converted_CCPP.csv')

# Select features to normalize
features_to_normalize = ['AT', 'V', 'AP', 'RH']

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Save the normalized dataset
data.to_csv('normalized_CCPP.csv', index=False)

# Print the first few rows to verify
print(data.head())

# Load the normalized dataset
data = pd.read_csv('normalized_CCPP.csv')

# Create composite features by multiplying normalized values
data['AT_V'] = data['AT'] * data['V']
data['AP_RH'] = data['AP'] * data['RH']

# Save the dataset with the new composite features
data.to_csv('composite_features_CCPP.csv', index=False)

# Print the first few rows to verify
print(data.head())

# Save the final dataset with all features
data.to_csv('final_features_CCPP.csv', index=False)

# Optional: Print the first few rows to verify
print(data.head())

# Load the dataset with all features
data = pd.read_csv('composite_features_CCPP.csv')

# Identify the selected features based on EDA
selected_features = ['AT', 'V', 'AP', 'AT_V', 'AP_RH', 'PE_Label']

# Create a new DataFrame with the selected features
selected_data = data[selected_features]

# Save the selected features to a new CSV file
selected_data.to_csv('selected_features_CCPP.csv', index=False)

print(selected_data.head())

# Load dataset
data = pd.read_csv('converted_CCPP.csv')

# Define features and target
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE_Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
accuracy_model_1 = metrics.accuracy_score(y_test, y_pred)
print("Model 1 Accuracy:", accuracy_model_1)

# Load dataset
data = pd.read_csv('normalized_CCPP.csv')

# Define features and target
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE_Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
accuracy_model_2 = metrics.accuracy_score(y_test, y_pred)
print("Model 2 Accuracy:", accuracy_model_2)

# Load dataset
data = pd.read_csv('composite_features_CCPP.csv')

# Define features and target
X = data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AP_RH']]
y = data['PE_Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
accuracy_model_3 = metrics.accuracy_score(y_test, y_pred)
print("Model 3 Accuracy:", accuracy_model_3)

# Load dataset
data = pd.read_csv('selected_features_CCPP.csv')

# Define features and target
X = data[['AT', 'V', 'AP', 'AT_V', 'AP_RH']]
y = data['PE_Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
accuracy_model_4 = metrics.accuracy_score(y_test, y_pred)
print("Model 4 Accuracy:", accuracy_model_4)

# Load the original dataset before normalization or feature engineering
data = pd.read_csv('converted_CCPP.csv')  # This is the dataset with categorical labels added

# Step 1: Select the original relevant features
selected_features = ['AT', 'V', 'AP', 'PE_Label']  # Adjust as necessary

# Create the composite features based on the original features
data['AT_V'] = data['AT'] * data['V']
data['AP_RH'] = data['AP'] * data['RH']  # Assuming RH is also relevant

# Select the composite features you just created
composite_features = ['AT_V', 'AP_RH']

# Combine selected features with composite features
final_selected_features = selected_features[:-1] + composite_features + ['PE_Label']

# Create a new DataFrame with the selected and composite features
selected_data = data[final_selected_features]

# Save this DataFrame to a new CSV file
selected_data.to_csv('selected_converted_CCPP.csv', index=False)

# Optional: Print the first few rows to verify
print(selected_data.head())

# Load dataset
data = pd.read_csv('selected_converted_CCPP.csv')

# Define features and target
X = data.drop(columns=['PE_Label'])  # Use all selected and composite features
y = data['PE_Label']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
accuracy_model_5 = metrics.accuracy_score(y_test, y_pred)
print("Model 5 Accuracy:", accuracy_model_5)

print(f"Model 1 (No normalization, no composite features) Accuracy: {accuracy_model_1 * 100:.2f}%")
print(f"Model 2 (Normalization, no composite features) Accuracy: {accuracy_model_2 * 100:.2f}%")
print(f"Model 3 (Normalization, composite features) Accuracy: {accuracy_model_3 * 100:.2f}%")
print(f"Model 4 (Selected features with normalization) Accuracy: {accuracy_model_4 * 100:.2f}%")
print(f"Model 5 (Selected features without normalization) Accuracy: {accuracy_model_5 * 100:.2f}%")

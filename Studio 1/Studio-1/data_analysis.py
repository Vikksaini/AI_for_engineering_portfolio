import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('CCPP.csv')

print(data.head())

print(data.info)

print(data.describe())

print(data.isnull().sum())

data.dropna(inplace=True)

# Removing duplicate entries from the dataset
data_cleaned = data.drop_duplicates()

# Display the number of rows after removing duplicates
print(f"Removal of duplicates, number of rows left: {data_cleaned.shape[0]}")

# Creating box plots for each numerical column
numerical_columns = data_cleaned.select_dtypes(include=[np.number]).columns

'''for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_cleaned[col], vert=False)
    plt.title(f'Boxplot for {col}')
  plt.show() '''


def remove_outliers_iqr(data, columns, multiplier=1.5):
    for col in columns:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Remove outliers
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data


# Apply the function iteratively
for _ in range(2):  # You can increase the number of iterations if needed
    data_cleaned = remove_outliers_iqr(data_cleaned, numerical_columns)
# Display the number of rows after removing outliers
print(f"Number of rows after removing outliers: {data_cleaned.shape[0]}")

# Create box plots again to check the result after removing outliers
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_cleaned[col], vert=False)
    plt.title(f'Boxplot for {col} after removing outliers')
    plt.show()

# Checking for missing values
print("\nMissing values:")
print(data_cleaned.isnull().sum())
# putting in mean values if there are empty spaces
data_cleaned.fillna(data_cleaned.mean(), inplace=True)
# Checking once again after putting in values
print(data_cleaned.isnull().sum())

# Assuming 'PE' is the target variable in the dataset
target = 'PE'
predictors = [col for col in data_cleaned.columns if col != target]

print("Target Variable:", target)
print("Predictors:", predictors)

# Plot histograms for each predictor
for col in predictors:
    plt.figure(figsize=(10, 6))
    sns.distplot(data_cleaned[col], bins=30, color='blue')
    plt.title(f'Histogram for {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

print("\nSummary Statistics:")
print(data_cleaned.describe())

# Scatter plots for each predictor against the target variable
for col in predictors:
    plt.figure(figsize=(10, 6))
    plt.scatter(data_cleaned[col], data_cleaned[target], alpha=0.5, color='green')
    plt.title(f'Scatter plot: {col} vs {target}')
    plt.xlabel(col)
    plt.ylabel(target)
    plt.show()

# Calculate the absolute correlation matrix
corr = abs(data_cleaned.corr())

# Select only the lower triangle of the correlation matrix
lower_triangle = np.tril(corr, k=-1)

# Mask the upper triangle
mask = lower_triangle == 0

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(lower_triangle, center=0.5, cmap='coolwarm', annot=True, xticklabels=corr.index, yticklabels=corr.columns,
            cbar=True, linewidths=1, mask=mask)
plt.title('Correlation Heatmap: Lower Triangle')
plt.show()
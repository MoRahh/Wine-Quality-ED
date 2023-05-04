import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/winequality-red.csv', sep=';')

# Set the plot size
plt.figure(figsize=(8,6))

# Plot a histogram of the quality ratings
plt.hist(df['quality'], bins=6, color='purple')

# Add labels and a title to the plot
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Histogram of Wine Quality Ratings')

# Set the plot size
plt.figure(figsize=(8,6))

# Plot a scatter plot of alcohol vs. quality
plt.scatter(df['alcohol'], df['quality'], color='blue')

# Add labels and a title to the plot
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Scatter Plot of Alcohol vs. Wine Quality')

# Preprocess the data
print(df.isnull().sum())  # Check for missing values
df.dropna(inplace=True)  # Remove any rows that contain missing values
X = df.iloc[:, :-1].values  # Split the data into X and y
y = df.iloc[:, -1].values
scaler = StandardScaler()  # Scale the features
X = scaler.fit_transform(X)
df = pd.get_dummies(df, columns=['quality'])  # One-hot encode any categorical variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the preprocessed data and the sizes of the training and testing sets
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Display the first few rows of the training set
print("Training set:")
print(pd.DataFrame(X_train).head())

# Display the first few rows of the testing set
print("Testing set:")
print(pd.DataFrame(X_test).head())

# Display the plot
plt.show()

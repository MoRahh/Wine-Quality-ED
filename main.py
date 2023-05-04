import pandas as pd
import matplotlib.pyplot as plt


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


# Display the plot
plt.show()

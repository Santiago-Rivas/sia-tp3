import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("perceptron_errors.csv")

# Extract mean values and errors
k_error_mean = df['k_error_mean']
full_error_mean = df['full_error_mean']
k_error_error = df['k_error_error']
full_error_error = df['full_error_error']

# Create a scatter plot
plt.scatter(df.index, k_error_mean, label='Mean of k-fold cross-validation')
plt.scatter(df.index, full_error_mean, label='Mean of normal training')

# Add error bars
# plt.errorbar(df.index, full_error_mean, yerr=full_error_error, fmt='none', ecolor='gray', label='Errors')

# Add labels and legend
plt.xlabel('Row Number')
plt.ylabel('full_error_mean')
plt.legend()

# Show plot
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
# Replace 'filename.csv' with the name of your CSV file.
df = pd.read_csv('/tmp/foo.txt')
df = pd.read_csv('~/Downloads/AzureFunctionsInvocationTraceForTwoWeeksJan2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt')

# Replace 'column_name' with the name of the column you're interested in
column = df['func']

# Count the frequency of unique values in the column
# This is the number of times each function was invoked
counts = column.value_counts()
# breakpoint()

normed_counts = counts / counts.sum()
scaled_counts = normed_counts * counts

perc = 0.7
per_hour = counts.quantile(perc) / 24 / 14 # 14 days, 24 hours
print(f"{perc*100}% of functions are called called {per_hour} times per hour or less on avarage")
# perc = 0.999
# per_hour = normed_counts.quantile(perc) / 24 / 14 # 14 days, 24 hours
# print(f"{perc*100}% of functions are called called {per_hour} times per hour or less on avarage")
# breakpoint()

r = range(counts.min(), counts.max() + 1)
# Plot the histogram using Seaborn
sns.histplot(data=counts, bins=100) #, stat="percent")
# sns.histplot(data=normed_counts, bins=100, stat="percent", cumulative=True)

# Set titles and labels
plt.title('Histogram of Value Counts in Column')
plt.xlabel('Invocation frequency')
plt.ylabel('Number of functions')
# plt.yscale('log')
# plt.xlim(0, 100000)
# plt.ylim(0, 25)

# Show the plot
plt.show()


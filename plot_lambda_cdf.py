import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame with event names
data = pd.DataFrame({
    # 'event_names': ['func_a', 'func_b', 'func_a', 'func_c', 'func_a', 'func_b', 'func_d']
    'event_names': ['func_a', 'func_b', 'func_a', 'func_c', 'func_b', 'func_a', 'func-b', 'func_b', 'func_b', 'func_d']
})
data = pd.DataFrame()
data['event_names'] = pd.read_csv('~/Downloads/AzureFunctionsInvocationTraceForTwoWeeksJan2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt')['func']
# https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md

# Count the frequency of each unique event
event_counts = data['event_names'].value_counts()

# Sort the frequencies in descending order
sorted_event_counts = event_counts.sort_values(ascending=False)


# Create a mapping of function names to numbers starting with 1
func_to_number = {func: number for number, func in enumerate(sorted_event_counts.index, start=1)}
# Replace function names with numbers in the original DataFrame
data['event_numbers'] = data['event_names'].map(func_to_number)
# Create a mapping of function names to numbers starting with 1
number_to_percent = lambda number: number / len(sorted_event_counts) * 100
# Replace function names with numbers in the original DataFrame
data['event_numbers'] = data['event_numbers'].map(number_to_percent)
# Re-count the frequency of each unique event number
sorted_event_counts = data['event_numbers'].value_counts(sort=True)


# Calculate the cumulative percentage
cumulative_percentage = np.cumsum(sorted_event_counts) / sorted_event_counts.sum() * 100

# print useful datapoints
functions = 10 # in percent
difference = abs(cumulative_percentage.index - functions)
index = difference.argmin()
function_percent = cumulative_percentage.index[index]
requests_percent = cumulative_percentage.values[index]
print(f"{function_percent:.2f}% of functions account for {requests_percent:.2f}% of requests")
perc = 0.7
per_hour = event_counts.quantile(perc) / 24 / 14 # 14 days, 24 hours
print(f"{perc*100}% of functions are called called {per_hour} times per hour or less on avarage")

# Plotting the Pareto chart
fig, ax = plt.subplots()
ax.bar(sorted_event_counts.index, sorted_event_counts.values, color='blue', alpha=0.7, label='Event Count')
ax.set_ylabel('Function Call Count', color='blue')

# Creating a second y-axis to plot the cumulative percentage
ax2 = ax.twinx()
ax2.plot(cumulative_percentage.index, cumulative_percentage.values, color='red', marker='x', label='Cumulative %')
ax2.set_ylabel('Cumulative %', color='red')
ax2.set_ylim(0, 110)

# Showing the plot
plt.title('Pareto Chart of Function Calls')
plt.grid()
ax.set_xlabel('Functions (normed %, sorted by call count)')
fig.tight_layout()
plt.show()


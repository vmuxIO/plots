import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample DataFrame with event names
data = pd.DataFrame({
    # 'event_names': ['func_a', 'func_b', 'func_a', 'func_c', 'func_a', 'func_b', 'func_d']
    'event_names': ['func_a', 'func_b', 'func_a', 'func_c', 'func_b', 'func_a', 'func-b', 'func_b', 'func_b', 'func_d']
})
# data['event_names'] = pd.read_csv('~/Downloads/AzureFunctionsInvocationTraceForTwoWeeksJan2021/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt')['func']

# Count the frequency of each unique event
event_counts = data['event_names'].value_counts()

# Sort the frequencies in descending order
sorted_event_counts = event_counts.sort_values(ascending=False)


# Create a mapping of function names to numbers starting with 1
func_to_number = {func: number for number, func in enumerate(sorted_event_counts.index, start=1)}
# Replace function names with numbers in the original DataFrame
data['event_numbers'] = data['event_names'].map(func_to_number)
# Re-count the frequency of each unique event number
sorted_event_counts = data['event_numbers'].value_counts(sort=True)


# Calculate the cumulative percentage
cumulative_percentage = np.cumsum(sorted_event_counts) / sorted_event_counts.sum() * 100

# breakpoint()

# Plotting the Pareto chart
fig, ax = plt.subplots()
ax.bar(sorted_event_counts.index, sorted_event_counts.values, color='blue', alpha=0.7, label='Event Count')
ax.set_ylabel('Event Count', color='blue')

# Creating a second y-axis to plot the cumulative percentage
ax2 = ax.twinx()
ax2.plot(cumulative_percentage.index, cumulative_percentage.values, color='red', marker='o', label='Cumulative %')
ax2.set_ylabel('Cumulative %', color='red')

# Showing the plot
plt.title('Pareto Chart of Function Calls')
fig.tight_layout()
plt.show()


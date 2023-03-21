import pandas as pd

# Load the results into a DataFrame
df = pd.read_csv('results/results.csv')

# Print the mean and standard deviation of the rewards
print(df['r'].mean())
print(df['r'].std())

# Plot the learning curve
import matplotlib.pyplot as plt
plt.plot(df['r'])
plt.show()

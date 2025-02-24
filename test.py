import numpy as np
import matplotlib.pyplot as plt

# Given data array
data = np.array([1.93839360e-22, 2.55820072e-20, 2.61525073e-18, 7.95527472e-17,
       5.51474430e-15, 2.66612305e-13, 1.00267103e-11, 2.93791075e-10,
       6.70884489e-09, 1.19416351e-07, 1.65723116e-06, 1.79359130e-05,
       1.51435275e-04, 9.97840311e-04, 5.13351432e-03, 2.06294164e-02,
       6.47816900e-02, 1.58994821e-01, 3.04837954e-01, 4.55619553e-01,
       5.27677659e-01, 4.65983625e-01, 2.99752289e-01, 1.19038499e-01,
       1.71846292e-04])

# Create a figure with a specified size
plt.figure(figsize=(8, 4))

# Plot the data with markers
plt.plot(data, marker='o', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of the Given Array')
plt.grid(True)

# Show the plot
plt.show()
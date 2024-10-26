# room-10
Discover the Hidden Algorithm - Classify the New Machine

Classify a new machine based on its similarity to nearby
machines using the proximity of their behavior. The closer a
machine’s behavior is, the more influence it has on the new
classification.


This script calculates the Euclidean distance from the new machine to each existing machine and finds the one with the smallest distance to classify it accordingly.


import numpy as np

# Machine behavior data
data = np.array([[2, 4], [4, 6], [5, 9], [7, 10]])
labels = ['Normal', 'Warning', 'Maintenance Needed', 'Fault']

# New machine behavior
new_machine = np.array([5, 8])

# Calculate Euclidean distances from the new machine to all existing machines
distances = np.linalg.norm(data - new_machine, axis=1)

# Find the index of the nearest neighbor
nearest_index = np.argmin(distances)

# Classify the new machine based on the nearest neighbor
classification = labels[nearest_index]

print(f"The new machine is classified as: {classification}")

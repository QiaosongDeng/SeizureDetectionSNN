import matplotlib.pyplot as plt

# Test data: thresholds and corresponding F1 scores
data = [
    [0.1, 71.32], [0.15, 74.32], [0.05, 70.24], [0.2, 70.98], [0.25, 62.19],
    [0.17, 79.29], [0.22, 79.22], [0.21, 79.68], [0.23, 66.08], [0.19, 79.16], [0.18, 80.85]
]

# Sort data by threshold for plotting
data_sorted = sorted(data, key=lambda x: x[0])
thresholds = [i[0] for i in data_sorted]
f1_scores = [i[1] for i in data_sorted]

# Create figure
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, marker='o', linestyle='-', color='b')
plt.title('Impact of Delta Modulation Threshold on SNN Model F1 Score')
plt.xlabel('Threshold')
plt.ylabel('F1 Score (%)')
plt.grid(True)
plt.show()
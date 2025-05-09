import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import sys

def plot_datapoints(data):
    plt.figure()
    plt.title('Data Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()

def find_clusters_with_history(input_distances, linkage):
    clusters = {}
    merge_history = []
    row_index, col_index = -1, -1
    array = list(range(input_distances.shape[0]))
    clusters[0] = array.copy()

    cluster_map = {i: [i] for i in range(len(array))}
    current_cluster_id = len(array)

    for k in range(1, input_distances.shape[0]):
        min_val = sys.maxsize

        for i in range(input_distances.shape[0]):
            for j in range(i):  # lower triangle
                if input_distances[i][j] < min_val:
                    min_val = input_distances[i][j]
                    row_index, col_index = i, j

        for i in range(input_distances.shape[0]):
            if i != row_index and i != col_index:
                if linkage.lower() == "single":
                    temp = min(input_distances[row_index][i], input_distances[col_index][i])
                elif linkage.lower() == "complete":
                    temp = max(input_distances[row_index][i], input_distances[col_index][i])
                elif linkage.lower() == "average":
                    temp = (input_distances[row_index][i] + input_distances[col_index][i]) / 2
                else:
                    raise ValueError("Invalid linkage type")

                input_distances[col_index][i] = input_distances[i][col_index] = temp

        # merge history
        cluster1 = array[row_index]
        cluster2 = array[col_index]
        size = len(cluster_map[cluster1]) + len(cluster_map[cluster2])
        merge_history.append([cluster1, cluster2, min_val, size])

        # merge clusters
        cluster_map[current_cluster_id] = cluster_map[cluster1] + cluster_map[cluster2]
        del cluster_map[cluster1]
        del cluster_map[cluster2]

        for n in range(len(array)):
            if array[n] == cluster1 or array[n] == cluster2:
                array[n] = current_cluster_id
        current_cluster_id += 1
        clusters[k] = array.copy()

        input_distances[row_index, :] = sys.maxsize
        input_distances[:, row_index] = sys.maxsize

    return clusters, np.array(merge_history)

def hierarchical_clustering(data, linkage, no_of_clusters):
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#ff7f0e', '#8c564b', '#e377c2', '#7f7f7f']
    initial_distances = pairwise_distances(data, metric='euclidean')
    np.fill_diagonal(initial_distances, sys.maxsize)

    clusters, merge_history = find_clusters_with_history(initial_distances.copy(), linkage)
    iteration_number = data.shape[0] - no_of_clusters
    cluster_labels = clusters[iteration_number]
    unique_clusters = np.unique(cluster_labels)

    plt.figure()
    plt.title(f'Clusters with {linkage} linkage')
    plt.xlabel('X')
    plt.ylabel('Y')
    cluster_to_color = {}

    for i, cluster_id in enumerate(unique_clusters):
        cluster_to_color[cluster_id] = colors[i % len(colors)]
        indices = np.where(np.array(cluster_labels) == cluster_id)[0]
        for j in indices:
            plt.scatter(data[j, 0], data[j, 1], color=cluster_to_color[cluster_id], label=f'Cluster {i}' if j == indices[0] else "")
    plt.legend()
    plt.show()

    return merge_history, cluster_labels, cluster_to_color

def plot_dendrogram(merge_history, cluster_labels, cluster_to_color):
    from collections import defaultdict
    n = merge_history.shape[0] + 1
    cluster_positions = {i: i for i in range(n)}
    cluster_heights = {i: 0 for i in range(n)}
    cluster_members = {i: [i] for i in range(n)}
    cluster_colors = {}

    # assigning colors to leaf nodes based on cluster labels
    for i in range(n):
        cluster_colors[i] = cluster_to_color[cluster_labels[i]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Dendrogram")
    ax.set_xlabel("Cluster Index")
    ax.set_ylabel("Distance")

    for i, (c1, c2, dist, _) in enumerate(merge_history):
        c1, c2 = int(c1), int(c2)
        x1 = cluster_positions[c1]
        x2 = cluster_positions[c2]
        h1 = cluster_heights[c1]
        h2 = cluster_heights[c2]
        new_x = (x1 + x2) / 2

        members = cluster_members[c1] + cluster_members[c2]
        cluster_members[n + i] = members

        # common label among members to assign color
        member_labels = [cluster_labels[m] for m in members]
        try:
            dominant_label = max(set(member_labels), key=member_labels.count)
            color = cluster_to_color[dominant_label]
        except:
            color = 'k'

        # vertical lines
        ax.plot([x1, x1], [h1, dist], c=color)
        ax.plot([x2, x2], [h2, dist], c=color)
        # horizontal connector
        ax.plot([x1, x2], [dist, dist], c=color)

        cluster_positions[n + i] = new_x
        cluster_heights[n + i] = dist
        cluster_colors[n + i] = color

    # the total vertical line it cuts at this point tells the number of clusters present
    num_clusters = len(np.unique(cluster_labels))
    cut_index = -(num_clusters - 1)  # index from the end
    cut_height = merge_history[cut_index][2]
    ax.axhline(y=cut_height, color='gray', linestyle='dotted', linewidth=1.5)

    plt.show()

if __name__ == "__main__":

    np.random.seed(42)
    data = np.random.randn(15, 2)
    plot_datapoints(data)
    merge_history, final_labels, cluster_to_color = hierarchical_clustering(data, "average", 4) # single, complete, average
    plot_dendrogram(merge_history, final_labels, cluster_to_color)

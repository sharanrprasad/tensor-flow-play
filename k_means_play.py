# K-means first step is to initialise K random cluster centroids called μ1, μ2 ... μk. Second step is to assign all
# training examples (x1 ... xm) to their nearest cluster centroid. c_i contains the index of the cluster centroid ( 1
# to k) a given training example of index i called x_i is assigned to. μ_c_i contains the cluster to which x_i is
# assigned to.
# Step 3 is to move the cluster centroids to the mean of the training examples assigned to them. Say if
# x_1 is (2,3) and x_2 is (4,5) then average is (3,4)

# Cost function of K-means is - J(c1...cm, x1..xm) = 1/m Sum(i=0 to m):(μ_c_i - x_i)^2


# Step 2 and step 3 are already actively working towards reducing this cost function.
# We repeat step 2 and 3 until there is no further reduction in cost function.


import numpy as np


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###

    for i in range(X.shape[0]):
        closest = 0
        close_magnitude = np.linalg.norm(np.subtract(X[i], centroids[0]))
        for j in range(centroids.shape[0]):
            if j == 0:
                continue
            mag = np.linalg.norm(np.subtract(X[i], centroids[j]))
            if mag < close_magnitude:
                closest = j
                close_magnitude = mag
        idx[i] = closest
    ### END CODE HERE ###

    return idx


# print(find_closest_centroids(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0, 0, 0], [5, 5, 5]])))


# UNQ_C2
# GRADED FUNCTION: compute_centroids

def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    ### START CODE HERE ###
    for i in range(len(centroids)):
        filtered_x_indices = np.where(idx == i)
        print("hello", filtered_x_indices)
        filtered_x = X[filtered_x_indices]
        print(filtered_x)
        centroids[i] = np.mean(filtered_x, axis=0)

    ### END CODE HERE ##

    return centroids


print(compute_centroids(np.array([[1, 2], [3, 4]]), np.array([0, 0]), 1))

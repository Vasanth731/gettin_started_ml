# singular_value_decomposition
import numpy as np
import pandas as pd

def SVD(A):
    ATA = np.dot(A.T, A)
    AAT = np.dot(A, A.T)

    # getting the eigenvalues
    eigenvalues_ATA, U = np.linalg.eigh(ATA)
    eigenvalues_AAT, V = np.linalg.eigh(AAT)

    # sorting the eigenvalues in descending order
    U = U[:, ::-1]
    V = V[:, ::-1]

    # getting the singular values from eigenvalues
    S = np.sqrt(np.abs(eigenvalues_ATA))[::-1]

    # check for numerical stability
    tol = np.finfo(np.float64).eps
    nonzero_singular_values = S > tol * np.max(S)
    
    # Trim U, S, V to include only non-zero singular values
    U = U[:, nonzero_singular_values]
    S = S[nonzero_singular_values]
    V = V[:, :U.shape[1]]  # to make U and V have the same number of columns

    return U, S, V

# dataloader
data = pd.io.parsers.read_csv('data/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::', encoding='ISO-8859-1')
movie_data = pd.io.parsers.read_csv('data/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::', encoding='ISO-8859-1')

# (m×u) with rows as movies and columns as users
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)

U, S, V = SVD(A)

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
        movie_data[movie_data.movie_id == movie_id].title.values[0]))

    for id in top_indexes + 1:
        if not movie_data[movie_data.movie_id == id].empty:
            print(movie_data[movie_data.movie_id == id].title.values[0])
        else:
            print(f"Movie with ID {id} not found in movie_data.")

k = 50
movie_id = 527 
top_n = 10 # number of recommendations needed

sliced = V.T[:, :k] # representative data
indexes = top_cosine_similarity(sliced, movie_id, top_n)
print_similar_movies(movie_data, movie_id, indexes)

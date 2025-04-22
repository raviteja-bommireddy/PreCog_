import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF

def reduce_with_svd(cooc_matrix, dims=[50, 100, 200, 300], random_state=42):
    """
    Perform dimensionality reduction using Truncated SVD.
    
    Parameters:
        cooc_matrix (np.ndarray): The co-occurrence matrix (shape: vocab_size x vocab_size)
        dims (list): List of dimensions to reduce to.
        random_state (int): Random state for reproducibility.

    Returns:
        dict: A dictionary mapping dimension -> reduced matrix (N x d)
    """
    svd_results = {}
    for d in dims:
        svd = TruncatedSVD(n_components=d, random_state=random_state)
        svd_results[d] = svd.fit_transform(cooc_matrix)
    return svd_results

def reduce_with_nmf(cooc_matrix, dims=[50, 100, 200], random_state=42, max_iter=300):
    """
    Perform dimensionality reduction using Non-negative Matrix Factorization (NMF).
    
    Parameters:
        cooc_matrix (np.ndarray): The co-occurrence matrix (shape: vocab_size x vocab_size)
        dims (list): List of dimensions to reduce to.
        random_state (int): Random state for reproducibility.
        max_iter (int): Max iterations for NMF convergence.

    Returns:
        dict: A dictionary mapping dimension -> reduced matrix (N x d)
    """
    nmf_results = {}
    for d in dims:
        nmf = NMF(n_components=d, init='nndsvd', random_state=random_state, max_iter=max_iter)
        nmf_results[d] = nmf.fit_transform(cooc_matrix)
    return nmf_results

# --- Get user input ---
print("Choose dimensionality reduction method:")
print("1. Truncated SVD (recommended for dense or general-purpose)")
print("2. NMF (non-negative, interpretable features)")
choice = input("Enter 1 or 2: ")

try:
    dims_input = input("Enter dimensions to reduce to (comma-separated, e.g., 50,100,200): ")
    dims = [int(d.strip()) for d in dims_input.split(',')]
except Exception:
    print("Invalid input. Using default dimensions [50, 100, 200].")
    dims = [50, 100, 200]

# --- Apply method ---
if choice == '2':
    print("\nYou selected: NMF")
    # reduced_results = reduce_with_nmf(cooc_matrix, dims)
    print("\n If we have cooc_matrix, we can use NMF for dimensionality reduction.")
else:
    print("\nYou selected: Truncated SVD")
    # reduced_results = reduce_with_svd(cooc_matrix, dims)
    print("\n If we have cooc_matrix, we can use SVD for dimensionality reduction.")

print("\nDimensionality reduction complete. Use reduced_results[dim] to access reduced matrix.")

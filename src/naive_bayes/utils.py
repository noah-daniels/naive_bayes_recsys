import numpy as np
import scipy as sp


def validate_interaction_matrix(X: sp.sparse.csc_array):
    if not isinstance(X, sp.sparse.csc_array):
        raise TypeError("Input interaction matrix must be a SciPy CSC sparse array.")
    if not np.issubdtype(X.dtype, np.bool_):
        raise TypeError("Input array must contain only boolean values.")

    return X.astype(np.int32)


def sparse_reciprocal(X: sp.sparse.sparray):
    """
    Compute the reciprocals of the non-zero elements of a sparse array.
    """
    reciprocal = X.copy()
    reciprocal.data = 1 / reciprocal.data
    return reciprocal


def spread_rows(X: sp.sparse.csr_array, row_indices: np.ndarray, target_shape: tuple[int]):
    """
    Spread the rows of X according to row_indices to create a new sparse array of target_shape.
    """
    assert row_indices.max() < target_shape[0]
    assert X.shape[1] == target_shape[1]

    rows = np.repeat(row_indices, X.shape[1])
    cols = np.tile(np.arange(X.shape[1]), X.shape[0])
    data = X.ravel(order="C")

    return sp.sparse.csr_array((data, (rows, cols)), shape=target_shape)


def get_top_ranks(arr, k, axis=1):
    """
    Rank a (sparse) array along the given axis.
    Keep only top-k ranks.
    Ties are assigned the same max rank (max ranking)
    """

    def _rank_arr(arr, k):
        if len(arr) <= k:
            idx = np.arange(len(arr))
            ranks = sp.stats.rankdata(arr, method="max")
            return ranks, idx
        else:
            val = arr[np.argpartition(arr, k - 1)[k - 1]]
            idx = np.where(arr <= val)[0]
            ranks = sp.stats.rankdata(arr[idx], method="max")
            mask = ranks <= k
            return ranks[mask], idx[mask]

    if sp.sparse.isspmatrix(arr):
        raise ValueError("Sparse matrices are not supported, convert to sparse array first.")

    if axis >= arr.ndim or axis < 0:
        raise ValueError("Axis out of range")

    if arr.shape[axis] < k:
        raise ValueError("K cannot exceed axis size.")

    if arr.min() < 0:
        if sp.sparse.issparse(arr):
            arr.data -= arr.min()
        else:
            arr -= arr.min()
        # raise ValueError("Array must be non-negative")

    if sp.sparse.issparse(arr):
        if axis == 1:
            sarr = arr.tocsr()
        else:
            sarr = arr.tocsc()

        new_data = []
        new_indices = []
        new_indptr = [0]
        for i in range(len(sarr.indptr) - 1):
            start, end = sarr.indptr[i], sarr.indptr[i + 1]
            data = sarr.data[start:end]
            indices = sarr.indices[start:end]

            if len(data) == 0:
                new_indptr.append(new_indptr[-1])
            else:
                a, b = _rank_arr(-data, k)
                new_data.extend(a)
                new_indices.extend(indices[b])
                new_indptr.append(len(new_data))

        if axis == 1:
            return sp.sparse.csr_array((new_data, new_indices, new_indptr), shape=sarr.shape)
        else:
            return sp.sparse.csc_array((new_data, new_indices, new_indptr), shape=sarr.shape)
    else:
        # indexing magic (thanks Gemini)
        # only works for max ranking
        idx = np.argpartition(-arr, k, axis=axis)
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(0, k + 1)
        idx_subset = idx[tuple(slc)]
        subset = np.take_along_axis(-arr, idx_subset, axis=axis)
        subset_ranks = sp.stats.rankdata(subset, method="max", axis=axis)
        result = np.zeros_like(arr, dtype=np.int32)
        np.put_along_axis(result, idx_subset, subset_ranks, axis=axis)
        result[result > k] = 0
        return result


def mask_matrix_rowwise(X, rows):
    mask = np.zeros(X.shape[0])
    mask[rows] = 1
    return sp.sparse.diags_array(mask).tocsr() @ X


def mask_matrix_colwise(X, cols):
    mask = np.zeros(X.shape[0])
    mask[cols] = 1
    return X @ sp.sparse.diags_array(mask).tocsc()

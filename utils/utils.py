import os
import errno

import numpy as np


def mkdir_p(path):
    """
    Make a directory.

    Parameters
    ----------
    path : str
        Path to the directory to make.

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def tgt_leq_tgt(time):
    """
    Lower triangular matrix where A_ij = 1 if t_i leq t_j.

    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.

    Returns
    -------
    tril: ndarray
        Lower triangular matrix.
    """
    t_i = time.astype(np.float32).reshape(1, -1)
    t_j = time.astype(np.float32).reshape(-1, 1)
    tril = np.where(t_i <= t_j, 1., 0.).astype(np.float32)
    return tril


def tgt_equal_tgt(time):
    """
    Used for tied times. Returns a diagonal by block matrix.
    Diagonal blocks of 1 if same time.
    Sorted over time. A_ij = i if t_i == t_j.

    Parameters
    ----------
    time: ndarray
        Time sorted in ascending order.

    Returns
    -------
    tied_matrix: ndarray
        Diagonal by block matrix.
    """
    t_i = time.astype(np.float32).reshape(1, -1)
    t_j = time.astype(np.float32).reshape(-1, 1)
    tied_matrix = np.where(t_i == t_j, 1., 0.).astype(np.float32)

    assert(tied_matrix.ndim == 2)
    block_sizes = np.sum(tied_matrix, axis=1)
    block_index = np.sum(tied_matrix - np.triu(tied_matrix), axis=1)

    tied_matrix = tied_matrix * (block_index / block_sizes)[:, np.newaxis]
    return tied_matrix


def iterate_minibatches(data, batchsize=32, shuffle=False):
    """
    Iterate minibatches.

    Parameters
    ----------
    data : ndarray
        Dataset to iterate over.
    batchsize : int
        Batch size. Default: 32.
    shuffle : bool
        Whether to shuffle the data before iterating over ot not.
        Default: False.

    Returns
    -------
    ndarray
        Yield minibatches.
    """
    if shuffle:
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]

    for start_idx in range(0, data.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield data[excerpt]

    if start_idx + batchsize != data.shape[0]:
        excerpt = slice(start_idx + batchsize, data.shape[0])
        yield data[excerpt]

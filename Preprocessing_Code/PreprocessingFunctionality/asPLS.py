import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

def csc_diff(x, n):
    '''Emulates np.diff(x,n) for a sparse matrix by iteratively taking difference of order 1'''
    assert isinstance(x, sp.sparse.csc_matrix) or (isinstance(x, np.ndarray) & len(x.shape) == 2), "Input matrix must be a 2D np.ndarray or csc_matrix."
    assert isinstance(n, int) & n >= 0, "Integer n must be larger or equal to 0."
    
    if n >= x.shape[1]:
        return sp.sparse.csc_matrix(([], ([], [])), shape=(x.shape[0], 0))
    
    if isinstance(x, np.ndarray):
        x = sp.sparse.csc_matrix(x)
        
    # set-up of data/indices via column-wise difference
    if(n > 0):
        for k in range(1,n+1):
            # extract data/indices of non-zero entries of (current) sparse matrix
            M, N = x.shape
            idx, idy = x.nonzero()
            dat = x.data
        
            # difference: this row (y) * (-1) + next row (y+1)
            idx = np.concatenate((idx, idx))
            idy = np.concatenate((idy, idy-1))
            dat = np.concatenate(((-1)*dat, dat))
            
            # filter valid indices
            validInd = (0<=idy) & (idy<N-1)

            # x_diff: csc_matrix emulating np.diff(x,1)'s output'
            x_diff =  sp.sparse.csc_matrix((dat[validInd], (idx[validInd], idy[validInd])), shape=(M, N-1))
            x_diff.sum_duplicates()
            
            x = x_diff

    return x

def asPLS(spectrum:np.ndarray, gamma:float = 1e5, min_error:float = 1e-5, max_iter:int = 1e4) -> np.ndarray:
    k = 2
    if spectrum.shape[0] == 1:
        spectrum = spectrum.transpose()
    spectrum = spectrum.flatten()
    
    N = len(spectrum)
    omega = sp.sparse.eye(N)
    alpha = sp.sparse.eye(N)
    D = csc_diff(sp.sparse.csc_matrix(sp.sparse.eye(N)), 2)
    D2 = D*D.transpose()

    baseline_prev = np.copy(spectrum)
    error = float('inf')
    iter = 0
    while iter <= max_iter:
        A = omega + gamma*alpha*D2
        b = omega*spectrum
        baseline = sp.sparse.linalg.spsolve(A, b)

        diff_spec_base = spectrum - baseline
        diff_baseline = baseline - baseline_prev
        error = np.linalg.norm(diff_baseline)
        if error <= min_error:
            break
        baseline_prev = np.copy(baseline)

        diff_spec_base_neg = (diff_spec_base < 0)*diff_spec_base
        std_dsbn = np.std(diff_spec_base_neg)
        alpha = sp.sparse.spdiags(np.abs(diff_spec_base), 0, N, N)/max(np.abs(diff_spec_base))
        omega = sp.sparse.spdiags(sp.special.expit(-k*(diff_spec_base - std_dsbn)/(std_dsbn)), 0, N, N)

        iter += 1
        if iter%100 == 0:
            print(f'current iteration: {iter}; \t current error: {error}')
    
    print(f'final iteration: {iter}; \t final error: {error}')
    return baseline


if __name__ == "__main__":
    baseline = asPLS(np.ones((40, )))

    plt.plot(baseline)
    plt.show()
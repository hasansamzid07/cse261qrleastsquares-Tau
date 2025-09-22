import numpy as np

def qr_decomposition(A):
    """Performs QR decomposition using Gram-Schmidt process."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R


if __name__ == "__main__":
    # Example
    A = np.array([[1, 1], [1, -1], [1, 1]])
    Q, R = qr_decomposition(A)
    print("Matrix A:\n", A)
    print("Q:\n", Q)
    print("R:\n", R)
    print("Check A = QR:\n", np.allclose(A, Q @ R))

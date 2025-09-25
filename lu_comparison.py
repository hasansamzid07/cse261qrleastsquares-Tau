import numpy as np
import matplotlib.pyplot as plt
from leastsquares import least_squares_qr

def least_squares_lu(A, b):
    """Solves least squares using normal equations (LU-based)."""
    AtA = A.T @ A
    Atb = A.T @ b
    return np.linalg.solve(AtA, Atb)


if __name__ == "__main__":
    # Generate synthetic ill-conditioned problem
    np.random.seed(0)
    x = np.linspace(0, 10, 20)
    y = 0.5 * x + 1 + np.random.normal(0, 0.5, size=x.shape)

    A = np.vstack([x, np.ones(len(x))]).T
    b = y

    # Solve using QR and LU
    coeffs_qr = least_squares_qr(A, b)
    coeffs_lu = least_squares_lu(A, b)

    print("QR Solution:", coeffs_qr)
    print("LU Solution:", coeffs_lu)

    # Stability test: perturb matrix
    errors_qr, errors_lu = [], []
    for eps in np.logspace(-10, -1, 10):
        A_perturbed = A + eps * np.random.randn(*A.shape)
        b_perturbed = b + eps * np.random.randn(*b.shape)

        sol_qr = least_squares_qr(A_perturbed, b_perturbed)
        sol_lu = least_squares_lu(A_perturbed, b_perturbed)

        errors_qr.append(np.linalg.norm(sol_qr - coeffs_qr))
        errors_lu.append(np.linalg.norm(sol_lu - coeffs_lu))

    plt.loglog(np.logspace(-10, -1, 10), errors_qr, label="QR Error")
    plt.loglog(np.logspace(-10, -1, 10), errors_lu, label="LU Error")
    plt.xlabel("Perturbation size (eps)")
    plt.ylabel("Error norm")
    plt.title("Stability Comparison: QR vs LU")
    plt.legend()
    plt.savefig("plots/error_analysis.png")
    plt.show()










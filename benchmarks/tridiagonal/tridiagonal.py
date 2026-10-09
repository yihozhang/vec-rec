import numpy as np


def thomas_rational_optimized(a, b, c, d):
    """
    Division-free forward Thomas algorithm with Nc eliminated.

    Parameters:
        a : subdiagonal (length n-1)
        b : main diagonal (length n)
        c : superdiagonal (length n-1)
        d : right-hand side (length n)
    """
    n = len(b)

    Nd = np.zeros(n, dtype=object)  # Numerators for d'
    D  = np.zeros(n, dtype=object)  # Common denominators D_i

    # Base case i = 0
    D[0] = b[0]
    Nd[0] = d[0]

    if n > 1:
        # Step i = 1 (uses D[-1] = 1 implicitly)
        D[1] = b[1] * D[0] - a[0] * c[0]
        Nd[1] = d[1] * D[0] - a[0] * Nd[0]

    # Forward pass: purely standalone 3-term recurrence for D and Nd
    for i in range(2, n):
        D[i]  = b[i] * D[i - 1] - (a[i - 1] * c[i - 1]) * D[i - 2]
        Nd[i] = d[i] * D[i - 1] - a[i - 1] * Nd[i - 1]

    # Back-substitution
    x = np.zeros(n, dtype=float)
    x[-1] = Nd[-1] / D[-1]

    for i in range(n - 2, -1, -1):
        # Using Nc_i = c_i * D_{i-1}
        D_prev = D[i - 1] if i > 0 else 1
        x[i] = (Nd[i] - c[i] * D_prev * x[i + 1]) / D[i]

    return x

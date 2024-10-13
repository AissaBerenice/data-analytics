
import numpy as np

def jacobi(A, b, X0, tol, N):
    n = len(b)
    X = X0.copy()
    for k in range(N):
        X_new = np.zeros_like(X)
        for i in range(n):
            s = sum(A[i][j] * X[j] for j in range(n) if j != i)
            X_new[i] = (b[i] - s) / A[i][i]
        
        # Check for convergence
        if np.linalg.norm(X_new - X, ord=np.inf) < tol:
            return X_new, k+1  # Return solution and iteration count

        X = X_new

    return None, N  # Return None if maximum iterations exceeded

# Define the matrix A and vector b for the given system of equations
A = np.array([[10, -1, 2, 0], 
              [1, 11, -1, 3], 
              [2, -1, 10, -1], 
              [0, 3, -1, 8]], dtype=float)

b = np.array([6, 25, -11, 15], dtype=float)

# Initial guess
X0 = np.zeros_like(b)

# Tolerance and maximum iterations
tol = 1e-6
N = 100

# Run the Jacobi iterative method
solution, iterations = jacobi(A, b, X0, tol, N)

if solution is not None:
    print(f"Solution: {solution} after {iterations} iterations")
else:
    print("Maximum iterations exceeded")

import numpy as np
def householder_qr(A):
	"""Compute the QR decomposition of matrix A using Householder reflections."""
	A = A.copy()
	m, n = A.shape
	Q = np.eye(m)
	
	for k in range(n):
		# Extract the vector to reflect
		x = A[k:, k]
		e = np.zeros_like(x)
		e[0] = np.linalg.norm(x)
		# Compute the Householder vector
		u = x - e
		v = u / np.linalg.norm(u) if np.linalg.norm(u) != 0 else u
		
		# Householder reflection matrix
		Hk = np.eye(m)
		Hk_sub = np.eye(len(x)) - 2.0 * np.outer(v, v)
		Hk[k:, k:] = Hk_sub
		
		# Apply reflection to A and accumulate Q
		A = Hk @ A
		Q = Q @ Hk

	R = A
	return Q, R

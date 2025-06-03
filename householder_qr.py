import numpy as np
import scipy

def diag_sign(A):
	"Compute the signs of the diagonal of matrix A"

	D = np.diag(np.sign(np.diag(A)))

	return D

def adjust_sign(Q, R):
	"""
	Adjust the signs of the columns in Q and rows in R to
	impose positive diagonal of Q
	"""

	D = diag_sign(Q)

	Q[:, :] = Q @ D
	R[:, :] = D @ R

	return Q, R


def householder_qr(A):
	return scipy.linalg.qr(A)
	n, m = A.shape # get the shape of A

	Q = np.empty((n, n)) # initialize matrix Q
	u = np.empty((n, n)) # initialize matrix u

	u[:, 0] = A[:, 0]
	Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

	for i in range(1, n):

		u[:, i] = A[:, i]
		for j in range(i):
			u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j] # get each u vector

		Q[:, i] = u[:, i] / np.linalg.norm(u[:, i]) # compute each e vetor

	R = np.zeros((n, m))
	for i in range(n):
		for j in range(i, m):
			R[i, j] = A[:, j] @ Q[:, i]

	return adjust_sign(Q, R)

A = np.arange(4*4).reshape(4, 4).astype(np.float32)
qr_mine = householder_qr(A)
qr_np = np.linalg.qr(A)
print(np.allclose(qr_mine[0], qr_np[0]), np.allclose(qr_mine[1], qr_np[1]) )
print(qr_mine)
print(qr_np)

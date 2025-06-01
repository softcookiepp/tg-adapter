import tinygrad
from .utils import is_jitted
import numpy as np
from .lambdas import diag, check_2d
from .tensor import convert_to_torch, convert_to_tg


def norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	raise NotImplementedError

def svd(A, full_matrices = True, driver = None, out = None):
	raise NotImplementedError
	A = A.tg
	ATA = A.T @ A
	
def qr(A):
	A = A.tg
	n, m = A.shape
	Q = tinygrad.Tensor.zeros((n, m), dtype=A.dtype, device = A.device)
	R = tinygrad.Tensor.zeros((m, m), dtype=A.dtype, device = A.device)
	for j in range(m):
		v = A[:, j]
		for i in range(j):
			R[i, j] = Q[:, i] @ A[:, j]
			v = v - R[i, j] * Q[:, i]
		R[j, j] = norm( convert_to_tg(v) ).tg
		Q[:, j] = v / R[j, j]
	return convert_to_torch(Q, R)
	
def eig(A, max_iter = 100, tol = 1e-6, out = None):
	# gotta see if jit is being used
	using_jit = is_jitted()
	
	A = A.tg
	n = A.shape[0]
	V = tinygrad.Tensor.eye(n, dtype = A.dtype, device = A.device)
	for _ in range(max_iter):
		Q, R = convert_to_tg( qr(convert_to_torch(A) ) )
		A = R @ Q
		V = V @ Q
		off_diag = (A - diag(diag(A)) ).abs()
		if (not using_jit) and off_diag.max() < tol:
			# can't check booleans when jit is being used
			break
	
	eigenvalues = diag(A)
	return convert_to_torch(eigenvalues, V)


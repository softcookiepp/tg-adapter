import tinygrad
from .utils import is_jitted
import numpy as np
from .lambdas import diag, check_2d, _outer
from .tensor import convert_to_torch, convert_to_tg
from .return_types import linalg_eig

import tinybloat


def norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	out = convert_to_torch(tinybloat.linalg.norm(A.tg, ord, dim, keepdim, out, dtype) )
	if not dtype is None:
		out = out.to(dtype)
	return out

def svd(A, full_matrices = True, driver = None, out = None):
	raise NotImplementedError
	A = A.tg
	ATA = A.T @ A
	
	
def _householder(x: tinygrad.Tensor):
	# adapted from here: https://stackoverflow.com/questions/53489237/how-can-you-implement-householder-based-qr-decomposition-in-python
	# archive: 
	alpha = x[0]
	s = tinybloat.linalg.norm(x[1:]) ** 2
	
	# tinygrad has no copy operator...
	v = x.zeros_like().contiguous()
	v[:] = x[:]
	
	if is_jitted():
		raise NotImplementedError("we flow control here for now")
	if s.numpy().item() == 0:
		tau = 0
	else:
		t = (alpha**2 + s)**0.5
		v[0] = alpha - t if alpha.numpy() <= 0 else -s / (alpha + t)
		v0s = v[0]**2
		tau = 2*v0s / (s + v0s)
		v = v/v[0]
	return v, tau
	
def _householder_vectorized(a):
	v = a / (a[0] + tinybloat.linalg.norm(a).copysign(a[0]) )
	v[0] = 1
	tau = 2 / (v.T @ v)
	return v, tau
	
	
def householder_qr(A, mode):
	"""Compute the QR decomposition of matrix A using Householder reflections."""
	m, n = A.shape
	Q = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device).contiguous()
	
	for k in range(n):
		# Extract the vector to reflect
		x = A[k:, k]
		e = x.zeros_like().contiguous()
		e[0] = tinybloat.linalg.norm(x)
		# Compute the Householder vector
		u = x - e
		_norm_out = tinybloat.linalg.norm(u)
		v = u / _norm_out if _norm_out.numpy() != 0 else u
		
		# Householder reflection matrix
		Hk = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device).contiguous()
		Hk_sub = tinygrad.Tensor.eye(len(x), dtype = A.dtype, device = A.device) - 2.0 * _outer(v, v)
		Hk[k:, k:] = Hk_sub
		
		# Apply reflection to A and accumulate Q
		A = Hk @ A
		Q = Q @ Hk

	R = A
	return Q, R

def qr(A, mode = "reduced"):
	if True:
		# ITS BANDAID TIME
		assert not is_jitted()
		A_device = A.tg.device
		A = A.tg.numpy()
		QR = np.linalg.qr(A)
		Q = tinygrad.Tensor(QR.Q.astype(A.dtype), device = A_device)
		R = tinygrad.Tensor(QR.R.astype(A.dtype), device = A_device)
		return convert_to_torch(Q, R)
	else:
		return convert_to_torch(householder_qr(A.tg, mode) )
	
def eig(A, max_iter = 100, tol = 1e-6, out = None):
	eigenvalues, eigenvectors = tinybloat.linalg.eig(convert_to_tg(A), max_iter, tol)
	return linalg_eig(*convert_to_torch(eigenvalues, eigenvectors) )



import tinygrad
from .utils import is_jitted
import numpy as np
from .lambdas import diag, _diag, check_2d, _outer
from .tensor import convert_to_torch, convert_to_tg


def _norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	if (not ord is None) and ord != 2:
		raise NotImplementedError(f"Order not implemented for tg_adapter.linalg.norm: {ord}")
	return ( (A**2).sum(axis = dim, keepdim = keepdim) )**0.5

def norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	out = convert_to_torch(_norm(A.tg, ord, dim, keepdim, out, dtype) )
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
	s = _norm(x[1:]) ** 2
	
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
	v = a / (a[0] + _norm(a).copysign(a[0]) )
	v[0] = 1
	tau = 2 / (v.T @ v)
	return v, tau
	
	
def _qr(A, mode = "reduced"):
	if not mode is "reduced":
		raise NotImplementedError(f"mode not implemented for tg_adapter.linalg.qr: {mode}")
	if len(A.shape) > 2:
		raise ValueError(f"Expected A to be 1D or 2D, but got {len(A.shape)}D instead")
	elif len(A.shape) == 1:
		A = A.reshape(1, -1)
		
	m, n = A.shape
	
	
	R = A.zeros_like().contiguous()
	R[:] = A[:]
	
	Q = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device)
	
	for j in range(0, n):
		v, tau = _householder_vectorized(R[j:, j].reshape(-1, 1))
		H = tinygrad.Tensor.eye(m, dtype = A.dtype, device = A.device).contiguous()
		H[j:, j:] -= tau * (v @ v.T)
		R = H @ R
		Q = H @ Q
	return Q[:n].T, R[:n].triu()
	
def householder_qr(A):
	"""Compute the QR decomposition of matrix A using Householder reflections."""
	A = A.copy()
	m, n = A.shape
	Q = tinygrad.eye(m, dtype = A.dtype, device = A.device).contiguous()
	
	for k in range(n):
		# Extract the vector to reflect
		x = A[k:, k]
		e = x.zeros_like().contiguous()
		e[0] = _norm(x)
		# Compute the Householder vector
		u = x - e
		_norm_out = _norm(u)
		v = u / _norm_out if _norm_out.numpy() != 0 else u
		
		# Householder reflection matrix
		Hk = tinygrad.eye(m, dtype = A.dtype, device = A.device).contiguous()
		Hk_sub = np.eye(len(x)) - 2.0 * np.outer(v, v)
		Hk[k:, k:] = Hk_sub
		
		# Apply reflection to A and accumulate Q
		A = Hk @ A
		Q = Q @ Hk

	R = A
	return Q, R

def qr(A, mode = "reduced"):
	A = A.tg
	return convert_to_torch(_qr(A, mode) )
	
def eig(A, max_iter = 100, tol = 1e-6, out = None):
	# gotta see if jit is being used
	using_jit = is_jitted()
	
	A = A.tg
	n = A.shape[0]
	V = tinygrad.Tensor.eye(n, dtype = A.dtype, device = A.device)
	for _ in range(max_iter):
		Q, R = _qr(A)
		A = R @ Q
		V = V @ Q
		off_diag = (A - _diag(_diag(A)) ).abs()
		if (not using_jit) and off_diag.max().numpy() < tol:
			# can't check booleans when jit is being used
			break
	
	eigenvalues = _diag(A)
	return convert_to_torch(eigenvalues, V)
	



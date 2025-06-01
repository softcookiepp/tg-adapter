import tinygrad
from .utils import is_jitted
import numpy as np
from .lambdas import diag, check_2d



def norm(A, ord = None, dim = None, keepdim = False, out = None, dtype = None):
	raise NotImplementedError

def svd(A, full_matrices = True, driver = None, out = None):
	raise NotImplementedError
	A = A.tg
	ATA = A.T @ A
	
def eig(A, max_iter = 100, tol = 1e-6, out = None):
	raise NotImplementedError
	A = A.tg
	n = A.shape[0]
	V = tinygrad.Tensor.eye(n, dtype = A.dtype, device = A.device)
	for _ in range(max_iter):
		Q, R = qr(A)
		A = R @ Q
		V = V @ Q
		#off_diag = (A - ).abs()
	

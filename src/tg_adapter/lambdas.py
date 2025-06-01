from .tensor import AdapterTensor as T
from .tensor import convert_to_tg, convert_to_torch
from .tensor import assert_same_device

exp = lambda x: T( x.tg.exp() )

# non-unaries actually have to be a function because of
# same device checking
def pow(x, y):
	x, y = convert_to_tg(x, y)
	assert_same_device(x.device, x, y)
	return T( x.pow(y) )

sum = lambda x, axis: T( convert_to_tg(x).sum(axis) )
sin = lambda x: T( x.tg.sin() )
cos = lambda x: T( convert_to_tg(x).cos() )
tan = lambda x: T( convert_to_tg(x).tan() )

def mean(inp, dim = None, keepdim = False, dtype = None, out = None):
	inp = inp.tg
	out = inp.mean(axis = dim, keepdim = keepdim)
	out = T(out)
	if not dtype is None:
		out = out.to(dtype)
	return out
	
def var(inp, dim = None, keepdim = False, dtype = None, out = None):
	inp = inp.tg
	out = inp.var(axis = dim, keepdim = keepdim)
	out = T(out)
	if not dtype is None:
		out = out.to(dtype)
	return out

def min(x, dim = None, keepdim = False):
	return x.min(dim, keepdim)
	
def max(x, dim = None, keepdim = False):
	return x.max(dim, keepdim)

from .F import sigmoid


from .F import cumprod

def stack(tensors, dim = 0, out = None):
	assert out is None
	tbase = tensors[0].tg
	trest = convert_to_tg( tuple(tensors[1:]) )
	assert_same_device(tbase.device, trest)
	return convert_to_torch(tbase.stack(*trest, dim = dim) )
	
def svd_lowrank(A, q = 6, niter = 2, M = None):
	raise NotImplementedError
	
	
def check_2d(t):
	if len(t.shape) != 2:
		raise RuntimeError(f"2D tensor expected, got {len(t.shape)}D tensor instead")

def _slice_to_square(t, offset = 0):
	if len(t.shape) == 1:
		return t
	n = min(t.shape)
	if offset >= 0:
		return t[offset:n, offset:n]
	else:
		return t[0:offset, 0: offset]
	
	
def diag(t, **kwargs):
	offset = 0
	if "diagonal" in kwargs.keys():
		offset = kwargs["diagonal"]
	t = t.tg
	t = _slice_to_square(t, offset)
	e = tinygrad.Tensor.eye(t.shape[0], dtype = t.dtype, device = t.device)
	if len(t.shape) == 1:
		# make diagonal matrix from 1-D tensor
		out = t.expand( (t.shape[0], t.shape[0]) ) * e
		if offset != 0:
			# TODO: implement padding
			raise NotImplementedError
		return T(out)
	elif len(t.shape) == 2:
		# make 1-D array from 2-D tensor
		return T( (t*e).sum(0) )
	else:
		raise RuntimeError(f"Expected 2D or 1D tensor, but got {len(t.shape) }D instead.")

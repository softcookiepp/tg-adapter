from .tensor import AdapterTensor as T
from .tensor import convert_to_tg, convert_to_torch
from .tensor import assert_same_device
import tinygrad
import tinybloat
from . import return_types

exp = lambda x: T( x.tg.exp() )

# non-unaries actually have to be a function because of
# same device checking
def pow(x, y):
	x, y = convert_to_tg(x, y)
	assert_same_device(x.device, x, y)
	return T( x.pow(y) )

sum = lambda x, axis: T( convert_to_tg(x).sum(axis) )
sin = lambda x: x.sin()
cos = lambda x: x.cos()
tan = lambda x: x.tan()

def mean(inp, dim = None, keepdim = False, dtype = None, out = None):
	return inp.mean(dim, keepdim, dtype, out)
	
def var(inp, dim = None, keepdim = False, dtype = None, out = None):
	inp = inp.tg
	out = inp.var(axis = dim, keepdim = keepdim)
	out = T(out)
	if not dtype is None:
		out = out.to(dtype)
	return out

def _min(x, dim = None, keepdim = False):
	return x.min(dim, keepdim)
	
def _max(x, axis = None, dim = None, keepdim = False):
	print(x.tg.device)
	out = convert_to_torch(tinybloat.max(convert_to_tg(x), axis = axis, dim = dim, keepdim = keepdim) )
	if isinstance(out, tuple):
		return return_types.minmax(*out)
	return out
	
def argmax(x, axis = None, dim = None, keepdim = False):
	print(x.tg.device)
	return convert_to_torch(tinybloat.argmax(convert_to_tg(x), axis = axis, dim = dim, keepdim = keepdim) )

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
		return t[0:n - offset, offset:]
	else:
		return t[0 - offset:, 0: offset]

def diag(t, *args, **kwargs):
	d = T(tinybloat.diag(t.tg, *args, **kwargs) )
	return d


def _outer(u, v):
	assert len(u.shape) == len(v.shape) == 1, "Both supplied tensors must be 1D"
	u_expanded = u.reshape(-1, 1).expand(-1, v.shape[0])
	v_expanded = v.reshape(1, -1).expand(u.shape[0], -1)
	return u_expanded * v_expanded

def outer(inp, vec2):
	return T( _outer( *convert_to_tg(inp, vec2) ) )
	
def zeros_like(inp):
	return T(inp.tg.zeros_like() )
	

def cumsum(inp, dim = None):
	return inp.cumsum(dim)
	
def matmul(a, b):
	return a @ b
	
def rsqrt(inp, *args, out = None):
	return inp.rsqrt()

def norm(inp, p='fro', dim=None, keepdim=False, out=None, dtype=None):
	if p != 2:
		raise NotImplementedError
	return T(tinybloat.linalg.norm(inp, ord = p, dim = dim, keepdim = keepdim, out = out, dtype = dtype) )
	
from .F import tanh

def multinomial(inp, num_samples, replacement=False, *args, generator=None, out=None):
	return T(inp.tg.multinomial(num_samples = num_samples, replacement = replacement) )

def where(condition, inp, other, *args, out=None):
	return T(condition.tg.where(inp.tg, other.tg) )	

def split(inp, split_size_or_sections, dim = 0):
	return inp.split(split_size_or_sections, dim = dim)
	
def abs(inp, *args, **kwargs):
	return inp.abs(*args, **kwargs)

def all(inp, *args, **kwargs):
	return inp.all(*args, **kwargs)

def log(inp, *args, **kwargs):
	return inp.log(*args, **kwargs)

def full_like(inp, *args, **kwargs):
	return inp.full_like(*args, **kwargs)

def ones_like(inp, dtype = None):
	ones = inp.full_like(1.0)
	if dtype is None:
		return ones
	return ones.to(dtype)
	
def isin(elements, test_elements, *args, assume_unique=False, invert=False):
	elements, test_elements = convert_to_tg(elements, test_elements)
	return convert_to_torch(tinybloat.isin(elements, test_elements, *args, assume_unique=assume_unique, invert=invert) )

def polar(inp, angle, *args, out = None):
	inp, angle = convert_to_tg(inp, angle)
	return convert_to_torch(tinybloat.polar(inp, angle) )

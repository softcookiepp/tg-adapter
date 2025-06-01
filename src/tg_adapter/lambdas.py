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

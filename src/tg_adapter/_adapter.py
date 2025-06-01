from .backend_environment_config import *

from . import nn, F
from .generator import *
from .lambdas import *
from .types import _get_type, dtype, get_default_dtype, set_default_dtype, finfo, is_floating_point
from .tensor_constructors import *
from .layouts import *
from . import distributed
from . import version
from . import compiler
from . import cuda
from . import linalg

from .io import *

import tinygrad
from .tensor import AdapterTensor

# aliases to twist shit into working at least somewhat
FloatTensor = AdapterTensor
LongTensor = AdapterTensor
IntTensor = AdapterTensor
BoolTensor = AdapterTensor
Tensor = AdapterTensor


#def device(dev):
#	dev = dev.upper()
#	return dev

# TODO: make an actual device class, then rejigger the state dict loader to do some other stuff
from .device import device, set_default_device, get_default_device


def is_grad_enabled():
	# pretty sure this will work
	return not tinygrad.Tensor.no_grad

def is_tensor(a):
	return isinstance(a, AdapterTensor)

Size = tuple

__version__ = "2.6.0"

from .F import chunk, clamp, cat

from .utils import no_grad

def __getattr__(attr):
	val = _get_type(attr)
	if not val is None:
		return val
	raise AttributeError

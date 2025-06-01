import tinygrad
from .types import get_default_dtype, get_tgt, _get_type
from .tensor import AdapterTensor as AT
import numpy as np
from .backend_environment_config import torch_dev_to_tiny
from typing import Iterable
from .device import parse_device

def _convert_size(size):
	if len(size) == 0:
		return size
	if isinstance(size[0], tuple) or isinstance(size[0], list):
		size = size[0]
	return size

def ones(*size, out=None, dtype=None, layout = None, device=None, requires_grad=False): #layout=torch.strided,
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	size = _convert_size(size)
	return AT(tinygrad.Tensor.ones(*size, device = device, dtype = dtype) )

def zeros(*size, out=None, dtype=None, layout = None, device=None, requires_grad=False): #layout=torch.strided,
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	size = _convert_size(size)
	return AT(tinygrad.Tensor.zeros(*size, device = device, dtype = dtype) )


def arange(start, end = None, step=1, out=None, dtype=None, layout="torch.strided", device=None, requires_grad=False):
	if dtype is None:
		dtype = _get_type("int32")
	
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	if end is None:
		end = start
		start = 0
	
	return AT(tinygrad.Tensor.arange(start, end, step, dtype = dtype, device = device) )

def empty(size = None, out=None, dtype=None, layout="torch.strided", device=None,
		requires_grad=False, pin_memory=False, memory_format="torch.contiguous_format"):
	assert not size is None
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	size = _convert_size(size)
	return AT(tinygrad.Tensor.empty(*size, device = device, dtype = dtype) )

def full(size, fill_value, out=None, dtype=None, layout="torch.strided", device=None, requires_grad=False):
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	return AT(tinygrad.Tensor.full(size, fill_value, device = device, dtype = dtype) )

def tensor(data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
	# get_backend_override is done internally for this one
	return AT(data, dtype, device,
			requires_grad, pin_memory)

def linspace(start, end, steps, *, out=None, dtype=None,
		layout="torch.strided", device=None, requires_grad=False):
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	t = tinygrad.Tensor.linspace(start, end, steps, dtype = dtype, device = device)
	return AT(t)
	
def from_numpy(a: np.ndarray):
	return AT(a)
	

def randn(*size, generator=None, out=None, dtype=None, layout="torch.strided", device=None, requires_grad=False, pin_memory=False):
	if isinstance(size[0], Iterable):
		size = size[0]
	device = parse_device(device).tg
	dtype = get_tgt(dtype, device)
	t = tinygrad.Tensor.randn(*size, dtype = dtype, requires_grad = requires_grad, device = device)
	return AT(t)

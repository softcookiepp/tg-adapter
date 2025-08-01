import tinygrad
from ..tensor import AdapterTensor as AT
from ..tensor import convert_to_tg, convert_to_torch
from ..device import parse_device, get_default_device

def uniform_(tensor, a = 0.0, b = 1.0, generator = None):
	return tensor.uniform_(a, b, generator = generator)

uniform = uniform_

# hopefully this works
def normal_(tensor, mean = 0.0, std = 1.0, generator = None):
	tensor = convert_to_tg(tensor)
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape,
		mean = mean,
		std = std,
		requires_grad = tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device).to(tensor.device)
	return AT(tensor.assign(norm.cast(tensor.dtype)) )

normal = normal_

def trunc_normal_(tensor, mean = 0.0, std = 1.0, a = -2.0, b = 2.0, generator = None):
	tensor = tensor.tg
	if not generator is None:
		raise NotImplementedError
	norm = tinygrad.Tensor.normal(*tensor.shape,
		mean = mean,
		std = std,
		requires_grad = tensor.requires_grad,
		dtype = tensor.dtype,
		device = tensor.device).to(tensor.device)
	norm = norm.clamp(a, b)
	return AT(tensor.assign(norm) )

def constant_(tensor, val):
	tensor = tensor.tg
	full = tensor.full_like(val)
	return AT(tensor.assign(full) )
	

def xavier_uniform_(tensor, *args, **kwargs):
	tensor = tensor.tg
	new = tinygrad.Tensor.glorot_uniform(tensor.shape, device = tensor.device, dtype = tensor.dtype, requires_grad = tensor.requires_grad)
	return AT(tensor.assign(new) )

xavier_uniform = xavier_uniform_

def xavier_normal_(tensor, *args, **kwargs):
	raise NotImplementedError

xavier_normal = xavier_normal_

def kaiming_uniform_(tensor, a = 0.0, *args, **kwargs):
	tensor = convert_to_tg(tensor)
	norm = tinygrad.Tensor.kaiming_uniform(*tensor.shape,
		a = a, 
		dtype = tensor.dtype,
		device = tensor.device)
	return AT(tensor.assign(norm.cast(tensor.dtype)) )

kaiming_uniform = kaiming_uniform_

def kaiming_normal_(tensor, *args, **kwargs):
	tensor = convert_to_tg(tensor)
	norm = tinygrad.Tensor.kaiming_normal(*tensor.shape,
		a = a, 
		dtype = tensor.dtype,
		device = tensor.device)
	return AT(tensor.assign(norm.cast(tensor.dtype)) )

kaiming_normal = kaiming_normal_

def zeros_(tensor):
	tensor = convert_to_tg(tensor)
	return AT(tensor.assign(tensor.zeros_like() ) )

def _calculate_correct_fan(*args, **kwargs):
	raise NotImplementedError

def _calculate_fan_in_and_fan_out(*args, **kwargs):
	raise NotImplementedError

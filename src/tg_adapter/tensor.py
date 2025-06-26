import tinygrad
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import unwrap

from .device import device as Device
from .device import tg_device_supports_longlong
import inspect
import numpy as np
from .types import get_type_from_tg, get_tgt, convert_np_type_correctly, _get_type, is_floating_point, get_np_type_from_torch
from .types import dtype as dtype_class
from .backend_environment_config import *
from .debugging import maybe_realize
from .utils import is_jitted

from tinybloat import ComplexTensor, safety_functions
import tinybloat


def _parse_to_arguments(*args, **kwargs):
	assert len(args) > 0 or len(kwargs) > 0
	dtype = None
	device = None
	
	for arg in args:
		if isinstance(arg, dtype_class):
			dtype = arg
		elif isinstance(arg, str):
			device = Device(arg)
		elif isinstance(arg, Device):
			device = arg
	if "dtype" in kwargs.keys():
		dtype = kwargs["dtype"]
	if "device" in kwargs.keys():
		device = kwargs["device"]
		if isinstance(device, str):
			device = Device(device)
	assert device is None or isinstance(device, Device)
	return dtype, device

class AdapterTensor:
	def __init__(self, data, dtype = None, device = None,
			requires_grad = False, pin_memory = False):
		# pin memory is unused, but kept for compatibility
		# convert device, duh
		self._is_complex = False
		tg_device = None
		if device is None:
			# default to CPU, just like torch does
			device = "cpu"
		
		if isinstance(data, float) or isinstance(data, int) or isinstance(data, list):
			data = np.array(data)
			if not dtype is None:
				npt = get_np_type_from_torch(dtype)
				data = data.astype(npt)
			
		if isinstance(device, Device):
			tg_device = device.tg
		elif isinstance(device, str):
			device = Device(device)
			tg_device = device.tg
		tgt = get_tgt(dtype, tg_device)
		if isinstance(data, tinygrad.Tensor):
			self._tg = data
		elif isinstance(data, ComplexTensor):
			self._tg = data
			self._is_complex = True
		elif isinstance(data, np.ndarray):
			real, imag = convert_np_type_correctly(data, tg_device)
			if not imag is None:
				self._tg = ComplexTensor(real, imag, device = tg_device)
				self._is_complex = True
			else:
				self._tg = tinygrad.Tensor(real, device = tg_device)
		else:
			data, _ = convert_np_type_correctly(np.array(data), tg_device )
			self._tg = tinygrad.Tensor(data, device = tg_device, dtype = tgt)
		self._dtype = dtype
		assert not self._tg is None
		self._rebuild_dtype()
		assert is_dtype_supported(self._tg.dtype, self._tg.device)
		maybe_realize(self._tg)
	
	def _rebuild_dtype(self):
		#from_tg = _get_type(self._tg.dtype)
		if self._is_complex:
			self._dtype = get_type_from_tg(self._tg.real.dtype, self._tg.device.split(":")[0], None, True)
		else:
			self._dtype = get_type_from_tg(self._tg.dtype, self._tg.device.split(":")[0], None)
	
	@property
	def tg(self):
		return maybe_realize(self._tg)
	
	@property
	def tgt(self):
		return self._dtype.tgt(self.tg.device.split(":")[0] )
		
	@property
	def shape(self):
		return self.tg.shape
	
	def dim(self):
		return len(self.shape)
		
	def is_contiguous(self):
		return unwrap(self._tg.lazydata.st).contiguous
	
	def size(self, idx = None):
		if idx is None:
			return self.shape
		else:
			return self.shape[idx]
			
	def _make_tensor(self, inp):
		# Create other tensor capable of operating with this one
		if not isinstance(inp, AdapterTensor):
			inp = AdapterTensor(inp, device = self.device)
		return maybe_realize(inp)
	
	@property
	def real(self):
		if self._is_complex:
			return AdapterTensor(self._tg.real)
		return self
	
	@property
	def imag(self):
		if self._is_complex:
			return AdapterTensor(self._tg.imag)
		raise RuntimeError("Real tensors have no imaginary component")
		
	@property
	def ndim(self):
		return len(self.shape)
	
	@property
	def dtype(self):
		self._rebuild_dtype()
		return self._dtype
	
	@property
	def data(self):
		# no idea why this exists :c
		return self
	
	def fill_(self, value):
		self.tg.assign(self.tg.zeros_like() + value)
	
	def zero_(self):
		self.fill_(0)
	
	def normal_(self, mean = 0.0, std = 1.0, generator = None):
		tensor = convert_to_tg(self)
		if not generator is None:
			raise NotImplementedError
		norm = tinygrad.Tensor.normal(*tensor.shape,
			mean = mean,
			std = std,
			requires_grad = tensor.requires_grad,
			dtype = tensor.dtype,
			device = tensor.device)
		self.tg.replace(norm)
	
	def uniform_(self, a, b, generator = None):
		tensor = self.tg
		if not generator is None:
			raise NotImplementedError
		uni = tinygrad.Tensor.uniform(*tensor.shape, low = a, high = b,
			dtype = tensor.dtype, requires_grad = tensor.requires_grad,
			device = tensor.device)
		self.tg.replace(tensor)
		return self
	
	def flip(self, dims):
		return self._tg_override(dims)
		
	def pow(self, other):
		return self ** other
		
	def mean(self, dim = None, keepdim = False, dtype = None, out = None):
		inp = self.tg
		out = inp.mean(axis = dim, keepdim = keepdim)
		out = convert_to_torch(out)
		if not dtype is None:
			out = out.to(dtype)
		return out
		
	def rsqrt(self, *args, out = None):
		return self._tg_override()
	
	def sum(self, axis = None):
		return convert_to_torch( x.tg.sum(axis) )
		
	@property
	def tdtype(self):
		return self.tgt
	
	def cuda(device = None, non_blocking = False, memory_format = "torch.preserve_format"):
		if not device is None:
			raise NotImplementedError
		return self.to("cuda")
	
	def cpu(self, memory_format = "torch.preserve_format"):
		return self.to("cpu")
	
	def to(self, *args, **kwargs):
		assert len(args) > 0 or len(kwargs) > 0
		
		dtype, device = _parse_to_arguments(*args, **kwargs)
		
		new_tensor = self._tg
		if device is None:
			device = self.device
		old_type = self._tg.dtype
		# gonna rewrite a little here c:
		if not dtype is None:
			old_dtype = dtype.tgt(device.tg)
			new_tensor = self._tg.cast(old_dtype)
		if not device is None:
			if True:
				# So it needs to be able to determine which type to convert to first before casting.
				# How do we do that??
				if dtype is None:
					dtype = self.dtype
				old_supported_type = self.dtype.tgt(self.device.tg)
				new_supported_type = dtype.tgt(device.tg)
				# the question that must be answered is, does the old device support the new type?
				# or does it not?
				
				# if new device supports new type and new device supports old type
				if is_dtype_supported(new_supported_type, device.tg) \
						and is_dtype_supported(old_supported_type, device.tg):
					# move first, then cast
					new_tensor = new_tensor.to(device.tg).cast(old_supported_type)
				# new device should support new type, dtype.tgt takes care of that
				# if new device doesn't support old type
				elif (not is_dtype_supported(old_supported_type, device.tg) ) and is_dtype_supported(new_supported_type, self.device.tg):
					# cast first, then move
					new_tensor = new_tensor.cast(new_supported_type).to(device.tg)
				else:
					# can't cast to type that neither supports!
					raise ValueError
				
					
			else:
				# first ensure that the dtype is compatible with the device
				if dtype is None:
					dtype = self.dtype
				supported_type_old_device = self.dtype.tgt(self.device.tg)
				supported_type_new_device = dtype.tgt(device.tg)
				# ok, so the problem is that it works one way, but not the other :c
				
				new_tensor = new_tensor.cast(supported_type_old_device)
				# then move it to the new device
				new_tensor = new_tensor.to(device.tg).cast(supported_type_new_device)
		return convert_to_torch(new_tensor)
		
		if dtype is None and (not device is None):
			new_tensor = maybe_realize(self.tg.to(device.tg) )
		elif (not dtype is None) and device is None:
			new_tensor = maybe_realize(self.tg.cast(dtype.tgt(self.device.tg) ) )
		elif not (dtype is None or device is None):
			return convert_to_torch(self.tg.to(device.tg).cast(dtype.tgt(device.tg)) )
		assert not new_tensor is None
		return convert_to_torch(new_tensor)
	
	def _tg_cast_(self, dtype):
		new_tensor = self.tg.cast(dtype.tgt(self.device.tg) )
		self.tg.replace(new_tensor)
		
	def _recast_to_supported_type(self, dev) -> tinygrad.Tensor:
		# recasts inplace to supported dtype
		supported_type = self.dtype.tgt(dev.tg)
		return self._tg.cast(supported_type)
	
	#def sum(self, *args, **kwargs):
	#	raise NotImplementedError	
	
	def to_(self, *args, **kwargs):
		# inplace equivalent of to()
		# torch has no equivalent, but it is still necessary for
		# the Module class to() method, since everything is done inplace there
		
		assert len(args) > 0 or len(kwargs) > 0
		
		dtype, device = _parse_to_arguments(*args, **kwargs)
		if not dtype is None:
			self._tg_cast_(dtype)
		if not device is None:
			new_t = self._recast_to_supported_type(device)
			self._tg.replace(new_t)
			self._tg.to_(device.tg)
		maybe_realize(self.tg)
		
		# forgot, have to set the data type to the correct one afterwards...
		self._rebuild_dtype()
		
	@property
	def device(self):
		dev = tiny_dev_to_torch(self.tg.device)
		return Device(dev)
	
	def _tg_override(self, *args, **kwargs):
		# Method for automatically wrapping stuff coded in tinygrad so
		# stuff works correctly
		
		# this will be the function that gets wrapped
		tg_attr = inspect.stack()[1].function
		
		# convert everything back to tinygrad.Tensor temporarily
		tg_self = self.tg
		tg_args = convert_to_tg(args)
		tg_kwargs = convert_to_tg(kwargs)
		
		assert_same_device(self.tg.device, tg_args, tg_kwargs)
		
		if len(tg_kwargs) == 0:
			# fix for methods that don't support **kwargs
			output = tg_self.__getattribute__(tg_attr)(*tg_args)
		else:
			output = tg_self.__getattribute__(tg_attr)(*tg_args, **tg_kwargs)
		return convert_to_torch(output)
		
	def __array__(self):
		return self.numpy()
		
	def __bool__(self):
		if is_jitted():
			# TinyJit doesn't behave well with python's flow control
			raise RuntimeError("Tensor cannot be evaluated as boolean when using TinyJit.\n Your code must be refactored to avoid using python's built-in flow control.")
		return self.item() > 0
	
	def __eq__(self, other):
		return self._tg_override(other)
		
	def ne(self, other):
		original_device = self.tg.device
		other_device = self.tg.device
		if isinstance(other, AdapterTensor):
			other = other.tg
			other_device = other.device
		if tinybloat.compatibility.device_supports_longlong(other_device) and tinybloat.compatibility.device_supports_longlong(self.tg.device):
			return AdapterTensor(self.tg != other)
		else:
			if isinstance(other, tinygrad.Tensor):
				other = other.to("CPU")
			return AdapterTensor( (self.tg.to("CPU") != other).to(original_device) )
	
	def int(self):
		return self.to(_get_type("int32") )
		
	def long(self):
		return self.to(_get_type("int64"))
		
	def type_as(self, other):
		return self.to(other.dtype)
	
	def item(self):
		if self.numel() > 1:
			raise RuntimeError("item() for multi-element tensor is ambiguous")
		return self._tg_override()
		
	def nonzero(self):
		assert not is_jitted(), "Nonzero gives tensors with variable length, and thus must not be jitted"
		return convert_to_torch( tinybloat.nonzero(self.tg) )
		
	
	def __add__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
	
	def __radd__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
		
	def __sub__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
		
	def __rsub__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
	
	def __mul__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
	
	def __rmul__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
	
	def __truediv__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
	
	def __rtruediv__(self, other):
		other = self._move_to_same_device(other)
		return self._tg_override(other)
		
	def numpy(self):
		return self.tg.numpy()
	
	def _reimplement_exact(self, function, *args, **kwargs):
		newself, args, kwargs = convert_to_tg(self, args, kwargs)
		assert_same_device(newself.device, args, kwargs)
		return convert_to_torch(newself.__getattribute__(function)(*args, **kwargs) )
	
	def masked_fill(self, *args, **kwargs):
		if False:
			args, kwargs = convert_to_tg(args, kwargs)
			return convert_to_torch(convert_to_tg(self).masked_fill(*args, **kwargs) )
		else:
			return self._tg_override(*args, **kwargs)
	
	def max(self, dim = None, keepdim = False):
		return self._tg_override(axis = dim, keepdim = keepdim)
	
	def min(self, dim = None, keepdim = False):
		return self._tg_override(axis = dim, keepdim = keepdim)

	def argmax(self, *args, **kwargs):
		#print(args, kwargs)
		#input(self.shape)
		#input(self.tg.chunk(11, dim = 1)[0].argmax(**kwargs).realize() )
		#return convert_to_torch(self._tg.contiguous().argmax(*args, **kwargs) )
		return convert_to_torch(safety_functions.argmax(self.tg, *args, **kwargs) )
	
	def view(self, *shape):
		return self._tg_override(*shape)
	
	def transpose(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def reshape(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def cast(self, dtype):
		# is this even a torch function? I don't know :c
		return AdapterTensor(self.tg.cast(dtype.tgt(self.tg.device) ) )
	
	def expand(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def _move_to_same_device(self, *inp, dev = None):
		if len(inp) == 1:
			inp = inp[0]
		if isinstance(inp, AdapterTensor):
			if dev is None:
				dev = _decide_device(self, inp)
			# gotta do the inplace to
			self.to_(dev)
			return inp.to(dev)
		if isinstance(inp, tinygrad.Tensor):
			raise NotImplementedError
			#return AdapterTensor(inp, device = self)
		elif isinstance(inp, list) or isinstance(inp, tuple):
			new = []
			for item in inp:
				new.append(self._move_to_same_device(item, dev = dev) )
			if isinstance(inp, tuple):
				new = tuple(new)
				
			return new
		elif isinstance(inp, dict):
			for k, v in inp.items():
				inp[k] = self._move_to_same_device(v, dev = dev)
			return inp
		else:
			if hasattr(inp, "__dict__"):
				# treat as dictionary hehe
				new_dict = self._move_to_same_device(inp.__dict__, dev = dev)
				inp.__dict__.update(new_dict)
				return inp
			else:
				# inp is a primitive type
				return inp
	
	def __getitem__(self, *args, **kwargs):
		
		self_device = self.device
		d = None
		if not tg_device_supports_longlong(self._tg.device):
			# Temporarily move tensors to CPU for indexing if absolutely required
			d = "cpu"
		args, kwargs = self._move_to_same_device(args, kwargs, dev = d)
		out = self._tg_override(*args, **kwargs)
		out = self._move_to_same_device( out, dev = self_device)
		return out

	def __gt__(self, other):
		# TODO: write tests for this crap
		return self._tg_override(other)
		
		
	def __lt__(self, other):
		# TODO: write tests for this crap
		return self._tg_override(other)
	
	def __ge__(self, other):
		# TODO: write tests for this crap
		return self._tg_override(other)
		
	def __le__(self, other):
		return self._tg_override(other)
		
	def __pow__(self, other):
		return self._tg_override(other)
	
	def __matmul__(self, other):
		if self._is_complex or (isinstance(other, AdapterTensor) and other._is_complex):
			cself = self._tg
			cother = other._tg
			if not isinstance(cother, ComplexTensor):
				cother = ComplexTensor(cother)
			elif not isinstance(cself, ComplexTensor):
				cself = ComplexTensor(cself)
			return convert_to_torch(cself @ cother)
		else:
			return self._tg_override(other)
		
	def pad(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def float(self):
		return self.to( _get_type("float32") )
		
	def flatten(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def unsqueeze(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def is_floating_point(self):
		return is_floating_point(self)
		
	def contiguous(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def repeat(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def permute(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def chunk(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def squeeze(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def clamp(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
	
	def interpolate(self, *args, **kwargs):
		return self._tg_override(*args, **kwargs)
		
	def numel(self):
		return self._tg_override()
	
	def __len__(self):
		if len(self.shape) == 0:
			return 1
		return self.shape[0]

def _decide_device(a: AdapterTensor, b: AdapterTensor) -> Device:
	if a.numel() > b.numel():
		return a.device
	elif b.numel() > a.numel():
		return b.device
	elif a.device == Device("cpu"):
		return b.device
	return a.device

def assert_same_device(dev, *inp):
	dev = tinygrad.Device.canonicalize(dev)
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		assert dev == inp.tg.device
	if isinstance(inp, tinygrad.Tensor):
		assert dev == inp.device
	elif isinstance(inp, list) or isinstance(inp, tuple):
		for item in inp:
			assert_same_device(dev, item)
	elif isinstance(inp, dict):
		for k, v in inp.items():
			assert_same_device(dev, v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			assert_same_device(dev, inp.__dict__)
	
def convert_to_torch(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		maybe_realize(inp.tg)
		return inp
	if isinstance(inp, tinygrad.Tensor) or isinstance(inp, ComplexTensor):
		return AdapterTensor(inp)
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_torch(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_torch(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_torch(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp
	
def convert_to_tg(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		return maybe_realize(inp.tg)
	if isinstance(inp, tinygrad.Tensor):
		# do nothing
		return maybe_realize(inp)
	elif isinstance(inp, list) or isinstance(inp, tuple):
		new = []
		for item in inp:
			new.append(convert_to_tg(item) )
		if isinstance(inp, tuple):
			new = tuple(new)
		for elem in new:
			assert not isinstance(elem, AdapterTensor)
			
		return new
	elif isinstance(inp, dict):
		for k, v in inp.items():
			inp[k] = convert_to_tg(v)
		return inp
	else:
		if hasattr(inp, "__dict__"):
			# treat as dictionary hehe
			new_dict = convert_to_tg(inp.__dict__)
			inp.__dict__.update(new_dict)
			return inp
		else:
			# inp is a primitive type
			return inp
	
	
def recursive_realize(*inp):
	if len(inp) == 1:
		inp = inp[0]
	if isinstance(inp, AdapterTensor):
		inp.tg.realize()
	if isinstance(inp, tinygrad.Tensor):
		inp.realize()
	elif isinstance(inp, list) or isinstance(inp, tuple):
		for item in inp:
			recursive_realize(item)
	elif isinstance(inp, dict):
		for k, v in inp.items():
			recursive_realize(v)
	else:
		if hasattr(inp, "__dict__"):
			recursive_realize(inp.__dict__)

from .backend_environment_config import torch_dev_to_tiny, tiny_dev_to_torch
import tinygrad

def backend_from_device(dev_str):
	if dev_str is None:
		return None
	return dev_str.split(":")[0]


class device:
	default_device = None
	
	def __init__(self, name):
		if isinstance(name, device):
			# just update __dict__
			self.__dict__.update(name.__dict__)
		else:
			self._name = name
			self._idx = None
			if ":" in name:
				self._name, self._idx = tuple(name.split(":"))
				self._idx = int(self._idx)
		assert not "float" in self._name
	
	@property
	def type(self):
		# pretty sure this is how it is done
		return self._name
	
	@property
	def name(self):
		return self._name
	
	@property
	def idx(self):
		return self._idx
	
	@property
	def tg(self):
		# Tinygrad device corresponding to this one
		return torch_dev_to_tiny(self._name, self._idx)
		
	def __repr__(self):
		r = f"device(\"{self._name}"
		if not self._idx is None:
			r += f":{self._idx}"
		return r + "\")"
	def __eq__(self, other):
		if not isinstance(other, device):
			return False
		return other.name == self.name and other.idx == self.idx

# have to make the constructor outside the class itself :c
if device.default_device is None:
	device.default_device = device("cpu")

def set_default_device(dev):
	if isinstance(dev, str):
		dev = device(dev)
	elif not isinstance(dev, device):
		raise ValueError
	device.default_device = dev

def get_default_device():
	return device.default_device
	
def parse_device(dev):
	if dev is None:
		return device.default_device
	elif isinstance(dev, str):
		return device(dev)
	elif isinstance(dev, device):
		return dev
	raise ValueError
	
# Odd little backend-specific quirks that tinygrad doesn't account for
_tg_device_quirk_table = {}

def tg_device_supports_longlong(dev: str) -> bool:
	# The test will make a big enough tensor that requires long long to index
	# If a compile error is thrown, then we return false.
	# If it succeeds, we return true
	big_shp = (1, 77, 49408, 768)
	a = tinygrad.Tensor.randn(big_shp)
	try:
		a.realize()
	except tinygrad.device.CompileError:
		del a
		return False
	del a
	return True

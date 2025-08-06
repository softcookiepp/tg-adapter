import tinygrad
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict, torch_load
from tinygrad import dtypes
from typing import Iterable
import inspect
from ..device import device
from ..tensor import AdapterTensor as AT
from ..tensor import convert_to_torch, convert_to_tg, _parse_to_arguments, recursive_realize
from ..debugging import KEEP_INPUT_TENSORS, realize_module_status, get_realize_depth
from tinybloat.common import recursive_get_attribute
import itertools
import os
import tinybloat

REALIZE_DEPTH = get_realize_depth()

# adapter for https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class Module:
	def __init__(self, *args, **kwargs):
		self._train = True
		self._input_spec = None
	
	def add_module(self, name: str, module = None):
		assert not name in self.__dict__.keys() or self.__getattribute__(name) == module
		self.__dict__[name] = module
		self.__getattribute__(name)
	
	def apply(self, fn):
		# fn should be a function that accepts a single module as an argument
		fn(self)
		for _, module in self.named_modules():
			fn(module)
	
	def buffers(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				for b in v.buffers():
					yield b
			elif isinstance(v, tinygrad.Tensor):
				# generally no idea what criteria is supposed to be used for this :c
				pass
	
	def bfloat16(self):
		raise NotImplementedError

	def named_children(self):
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				yield k, v

	def children(self):
		# Return an iterator over immediate children modules.
		for _, v in self.named_children():
			yield v
	
	def get_paramteter(self, name: str):
		for k, v in self.named_parameters():
			if k == name:
				return v
		raise AttributeError(f"Could not find parameter {name}")
		
	
	def get_buffer(self, name: str):
		for k, v in self.named_parameters():
			if k == name:
				return v
		raise AttributeError(f"Could not find buffer {name}")
	
	def requires_grad_(self, *args, **kwargs):
		# tinygrad doesn't keep track of this lol
		pass
	
	def parameters(self, recurse = True):
		if not recurse:
			raise NotImplementedError
		#params = []
		for k, v in self.named_parameters():
			#params.append(v)
			yield v
		if isinstance(self, Iterable):
			for i, elem in enumerate(self):
				if hasattr(elem, "named_parameters"):
					for k, mod in elem.named_parameters():
						#params.append(mod)
						yield mod
		#return params
		"""
		params = []
		for k, v in enumerate(self):
			print(v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for k, v in self.__dict__.items():
			print(k, v)
			if isinstance(v, tinygrad.Tensor):
				params.append(v)
			elif isinstance(v, list):
				for item in v:
					if isinstance(v, tinygrad.Tensor):
						params.append(v)
		for child in self.children():
			for param in self.children.params():
				params.append(param)
		return params
		"""

	def compile(self):
		# tinygrad has no compile equivalent lol
		pass
		
	def to(self, *args, **kwargs):
		# convert any floating point parameters to dtype
		# move parameters to device, convert unsupported types to supported types if required
		# how do we do this?
		
		dtype, device = _parse_to_arguments(*args, **kwargs)
		if dtype is None and device is None:
			raise ValueError
		
		if not device is None:
			# why not just pre-move everything?
			# hopefully this works pls
			tinybloat.cast_to_supported_and_move_(self, device.tg)	
		
		assert dtype == None or dtype.is_floating_point
		for k, v in self.state_dict().items():
			if isinstance(v, AT) and v.is_floating_point():
				# typecast it
				v.to_(dtype, device)
			elif isinstance(v, AT) and not device is None:
				v.to_(device)
		return self
		
	def cuda(self, device = None):
		raise NotImplementedError
		
		
		
	def train(self, train = True):
		self._train = train
		
	def eval(self):
		self.train(False)
		
	@tinygrad.TinyJit
	def _jit_forward(self, *args, **kwargs):
		# Only used when a module is a root module; makes it a lot faster
		out = self.forward(*args, **kwargs)
		recursive_realize(out)
		return out
		
	def _run_forward(self, *args, **kwargs):
		if hasattr(self, "tg_forward"):
			# use tinygrad-based forward method instead if it exists
			args, kwargs = convert_to_tg(args, kwargs)
			out = self.tg_forward(self.tg, *args, **kwargs)
			return convert_to_torch(out)
		else:
			return self.forward(*args, **kwargs)
		
	def __call__(self, *args, **kwargs):
		# get parent function, check if it is not forward or __call__,
		# then invoke realize() if that is the case
		parent_function = inspect.stack()[1].function
		
		args, kwargs = convert_to_torch(args, kwargs)
		
		if self._is_submodule() and (not realize_module_status() ):
			out = self._run_forward(*args, **kwargs)
		else:
			# use jit if root module
			#out = self._jit_forward(*args, **kwargs)
			# actually disabling for now, it does funny stuff :c
			out = self._run_forward(*args, **kwargs)
			recursive_realize(out)
		
		# this is here for the submodule tester thingy
		if KEEP_INPUT_TENSORS:
			self._input_spec = [args, kwargs]
		
		return out
		
	def forward(self, *args, **kwargs):
		raise NotImplementedError
		
	@property
	def _modules(self):
		modules = {}
		# immediate modules
		for k, v in self.__dict__.items():
			if isinstance(v, Module):
				modules[k] = v
			# TODO: reimplement modulelist
		return modules
		
		
	def _get_p(self):
		d = {}
		for k, v in self.named_parameters():
			if "." in k:
				# member of submodule, don't use
				continue
			d[k] = v
			
		return d
		
	@property
	def _parameters(self):
		return self._get_p()
		
	def _named_members(self, get_members_fn):
		# huggingface really likes to touch privates
		return get_members_fn(self)
	
	@property
	def _buffers(self):
		return self._get_p()
	
	def _load_from_state_dict(self,
			state_dict,
			prefix,
			local_metadata,
			strict,
			missing_keys,
			unexpected_keys,
			error_msgs):
		# just because huggingface doesn't seem to understand that you REALLY SHOULD NOT BE USING PRIVATE METHODS
		# goodness gracious
		for k, v in self.__dict__.items():
			full_key = prefix + k
			if full_key in state_dict.keys():
				v.tg.replace(state_dict[full_key].to(v.tg.device) ).realize()
				print("initialized", full_key)
	
	def _load_elem_state_dict_recursive(self, k, v, state_dict, prefix):
		if isinstance(v, Module):
			v._load_state_dict_recursive(state_dict, prefix = prefix)
		elif isinstance(v, AT):

			new_key = prefix.strip(".")
			if new_key in state_dict.keys():
				tg_tensor = state_dict[new_key]
				if isinstance(state_dict[new_key], AT):
					if v.tg.shape == state_dict[new_key].tg.shape:
						v.tg.replace(state_dict[new_key].tg.to(v.tg.device) ).realize()
					else:
						# just do something risky lmao
						v._tg = state_dict[new_key].tg.to(v.tg.device)
						v.tg.realize()
				else:
					v.tg.replace(state_dict[new_key].to(v.tg.device) ).realize()
			else:
				# TODO: warn user or something, i forget
				pass
	
	def _load_state_dict_recursive(self, state_dict, prefix = ""):
		for k, v in self.__dict__.items():
			if isinstance(v, list):
				for i in range(len(v) ):
					vi = v[i]
					self._load_elem_state_dict_recursive(str(i), vi, state_dict, prefix = f"{prefix}.{k}.{i}")
			else:
				self._load_elem_state_dict_recursive(k, v, state_dict, f"{prefix}.{k}")
	
		
	def load_state_dict(self, state_dict, strict = True, assign = False, prefix = ""):
		"""
		for k, v in self.__dict__.items():
			if isinstance(v, list):
				for i in range(len(v) ):
					new_prefix = ".".join( [prefix, k, str(i)] )
		"""
		# actually keep doing it this way, using tinygrad's method is going to screw shit up once we get into ModuleList s
		self._load_state_dict_recursive(state_dict)
		return [], []
		
		
		#raise NotImplementedError
		#_disinherit(self)
		# use conventional method, but replace all dict keys with x._tg lol
		new_state_dict = {}
		for k, v in state_dict.items():
			k = k + "._tg"
			new_state_dict[k] = v
		tinygrad.nn.state.load_state_dict(self, new_state_dict, strict = strict, verbose = True)
		#_cb(self)
		# expected and missing keys are not implemented yet
		return [], []
		
	def tg_state_dict(self):
		# Returns the state dict in tinygrad format
		return convert_to_tg(self.state_dict() )
		
	
	def state_dict(self, prefix = ""):
		#return _disinherit(tinygrad.nn.state.get_state_dict(self) )
		# Can no longer do that, as AdapterTensor objects are no longer
		# a subclass of tinygrad.Tensor.
		# we will have to make a dedicated method...
		state_dict = {}
		for k, v in self.__dict__.items():
			if isinstance(v, list):
				for i in range(len(v) ):
					l_prefix = ".".join([prefix, f"{k}.{i}"])
					if isinstance(v[i], Module):
						state_dict.update(v[i].state_dict(l_prefix) )
					
			elif isinstance(v, Module):
				new_prefix = prefix + f".{k}"
				state_dict.update(v.state_dict(new_prefix) )
			elif isinstance(v, AT):
				sd_key = ".".join([prefix, k]).strip(".")
				state_dict[sd_key] = v
		return state_dict
				
	
	def __repr__(self):
		return f"{self.__class__}"
	
	def named_parameters(self, memo = None, prefix = "", remove_duplicate = True):
		for name, param in self.state_dict().items():
			yield name, param
	
	def get_submodule(self, target: str):
		for k, v in self.named_modules():
			if k == target:
				return v
		raise AttributeError(f"Module {target} not found")
	
	def named_modules(self, memo=None, prefix="", remove_duplicate=True):
		# So this is supposed to be recursive.
		# We can probably extract a good deal of it from the state dict
		# instead of doing the recursive crap
		sd = self.state_dict()
		named_modules = {}
		for k, v in sd.items():
			# iterate over each subkey and append it to the list of named modules
			subkeys = k.split(".")
			for ik in range(len(subkeys) ):
				subkey = ".".join(subkeys[0:ik+1])
				if not subkey in named_modules.keys():
					attr_value = recursive_get_attribute(self, subkey)
					#print(subkey, attr_value)
					if isinstance(attr_value, Module) or isinstance(attr_value, list):
						named_modules[subkey] = attr_value
		return itertools.chain([("", self)], named_modules.items() )
				
	def modules(self, remove_duplicate = True):
		for k, v in self.named_modules(remove_duplicate = remove_duplicate):
			yield v
	
	@property
	def tg(self):
		return AutogenTinygradModule(self)
	
	def _is_submodule(self):
		found_call = False
		found_forward = False
		forward_count = 0
		call_count = 0
		for item in inspect.stack()[2:]:
			if item.function == "forward":
				found_forward = True
				forward_count += 1
			elif item.function == "__call__" and os.path.basename(item.filename) == "module.py":
				found_call = True
				call_count += 1
		return forward_count > REALIZE_DEPTH and call_count > REALIZE_DEPTH
	
	def register_buffer(self, name, tensor, persistent = True):
		assert not name in self.__dict__.keys()
		self.__dict__[name] = tensor
	
	def register_parameter(self, name, tensor, persistent = True):
		assert not name in self.__dict__.keys()
		self.__dict__[name] = tensor
		
	@property
	def training(self):
		# TODO: actually implement
		return False


# This is the class that will be returned when accessing the Module.tg
# property.
# It exists to make it possible to port torch code to vanilla tinygrad
# step-by-step rather than all at once.
class AutogenTinygradModule:
	def __init__(self, tga_module: Module):
		for k, mod in tga_module.named_modules():
			assert hasattr(mod, "tg_forward"), "Module must have tg_forward method to be able to converted to a vanilla tinygrad model."
			
			# Now we just need a way of finding which modules are immediate...
			# oh right
			# count of 0 means not child of child, length means it is a child and not self
			if k.count(".") == 0 and len(k) > 0:
				# yes, this will be recursive c:
				self.__dict__[k] = mod.tg
		
		for k, param in tga_module.named_parameters():
			# ensure that param is immediate (not member of child module)
			if k.count(".") == 0:
				self.__dict__[k.split(".")[0]] = param.tg
				
		for k, v in tga_module.__dict__.items():
			if not k in self.__dict__.keys():
				self.__dict__[k] = v
		
		self.tg_forward = tga_module.tg_forward
	
	def __call__(self, *args, **kwargs):
		return self.tg_forward(self, *args, **kwargs)


# class for converting pure tinygrad modules to tg_adapter modules.
# This will be useful at some point, I absolutely know it.
class AutogenAdapterModule(Module):
	def __init__(self, tinygrad_module, *args, **kwargs):
		for k, v in tinygrad_module.__dict__.items():
			if is_tinygrad_module(v):
				self.__dict__[k] = AutogenAdapterModule(v)
			elif isinstance(v, tinygrad.Tensor):
				self.__dict__[k] = AT(v)
			else:
				# generic object/attribute
				self.__dict__[k] = v
		# that should do it pretty easily!
		# now we just need the __call__ method to be wrapped...
		self._original_call = tinygrad_module.__call__
		
	def forward(self, *args, **kwargs):
		raise NotImplementedError
	
	

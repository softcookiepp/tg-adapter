from .module import Module
import tinygrad
from tinygrad.helpers import make_tuple, prod
import math
from ..tensor import AdapterTensor as AT
from ..tensor import convert_to_torch, convert_to_tg, assert_same_device
from .. import tensor_constructors as tc
from . import init as internal_init
from ..types import highest_precision_int
from ..device import tg_device_supports_longlong
from .parameter import Parameter
from .. import tensor_constructors as tc

class AvgPool2d(Module):
	def __init__(self, kernel_size, stride=None, padding=0,
			ceil_mode=False, count_include_pad=True):
		super().__init__()
		self._kernel_size = kernel_size
		self._stride = stride
		if stride is None:
			self._stride = kernel_size
		self._padding = padding
		self._ceil_mode = ceil_mode
		self._count_include_pad = count_include_pad
	
	"""
	def forward(self, inp):
		inp = convert_to_tg(inp)
		out = inp.avg_pool2d(self._kernel_size, self._stride,
			1, self._padding, self._ceil_mode, self._count_include_pad)
		return AT(out)
	"""
	def tg_forward(_, self, inp):
		return inp.avg_pool2d(self._kernel_size, self._stride,
			1, self._padding, self._ceil_mode, self._count_include_pad)

class SequentialIterator:
	def __init__(self, module):
		self._elem = 0
		self._module = module
	
	def __next__(self):
		if self._elem < len(self._module.args):
			elem = self._module.args[elem]
			self._elem += 1
			return elem
		else:
			raise StopIteration

class Sequential(Module):
	def __init__(self, *args):
		super().__init__()
		self._args = args
	
	@property
	def args(self):
		return self._args
		
	def forward(self, x):
		for arg in self._args:
			x = arg(x)
		return x
		
	def __iter__(self):
		return SequentialIterator(self)
		
	
class Dropout(Module):
	def __init__(self, p = 0.5, inplace = False):
		if inplace:
			# tinygrad has no inplace operator for dropout and I feel way too lazy to make one
			raise NotImplementedError
		
		self._p = p
	"""
	def forward(self, inp):
		return AT(inp.tg.dropout(self._p) )
	"""
	def tg_forward(_, self, inp):
		return inp.dropout(self._p)

class AdaGroupNorm(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
		
class Identity(Module):
	def __init__(self, *args, **kwargs):
		pass
	def forward(self, *args, **kwargs):
		if len(args) == 1:
			return args[0]
		else:
			return args
		
class Upsample(Module):
	def __init__(self, *args, **kwargs):
		pass
	
	def forward(self, *args, **kwargs):
		raise NotImplementedError

class ConvNd(Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		
		# must have dimensionality
		assert not dim is None
		if isinstance(padding, str):
			assert padding in ["valid", "same"]
		self.kernel_size = make_tuple(kernel_size, dim)
		self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
		scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
		
		self._in_channels = in_channels
		self._out_channels = out_channels
		
		#self.weight = tinygrad.Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
		self.weight = tc.empty(  (out_channels, in_channels//groups, *self.kernel_size)  )
		internal_init.uniform_(self.weight, a = -scale, b = scale)
		#self.weight = AT(tinygrad.Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale) )
		
		self.bias = None
		if bias:
			self.bias = tc.empty(  (out_channels,)  )
			internal_init.uniform_(self.bias, a = -scale, b = scale)
			#self.bias = AT( tinygrad.Tensor.uniform(out_channels, low=-scale, high=scale) )
	
	@property
	def in_channels(self):
		return self._in_channels
	
	@property
	def out_channels(self):
		return self._out_channels
	
	"""
	def forward(self, x):
		raise RuntimeError("we aren't supposed to get here!")
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		x = x.conv2d(weight, bias, self.groups, self.stride, self.dilation, self.padding)
		return AT(x)
	"""
	
	def tg_forward(_, self, x):
		return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)

# ugh, I forgot that torch is going to expect this crap as a type :c

class Conv1d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 1)
	
class Conv2d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 2)

class Conv3d(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 3)


class ConvTransposeNd(ConvNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, output_padding=0, groups=1, bias=True, dilation=1,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		assert not dim is None
		super().__init__(self, in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim)
		scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
		self.weight = tc.empty(  (in_channels, out_channels//groups, *self.kernel_size)  )
		internal_init.uniform_(self.weight, a = -scale, b = scale)
		#self.weight = AT(tinygrad.Tensor.uniform(in_channels, out_channels//groups, *self.kernel_size, low=-scale, high=scale) )
		self.output_padding = output_padding
	
	def tg_forward(_, self, x):
		return x.conv_transpose2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding, self.output_padding)

class ConvTranspose1d(ConvTransposeNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 1)
	
class ConvTranspose2d(ConvTransposeNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 2)

class ConvTranspose3d(ConvTransposeNd):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
			padding=0, dilation=1, groups=1, bias=True,
			padding_mode='zeros', device=None, dtype=None, dim = None):
		super().__init__(in_channels, out_channels, kernel_size, stride,
			padding, dilation, groups, bias, padding_mode, device, dtype, dim = 3)
			
class LayerNorm(Module):
	def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True,
			bias=True, device=None, dtype=None):
		self.normalized_shape: tuple[int, ...] = make_tuple(normalized_shape, 1)
		self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
		#self.weight = AT(tinygrad.Tensor.ones(*self.normalized_shape) ) if elementwise_affine else None
		#self.bias = AT(tinygrad.Tensor.zeros(*self.normalized_shape) ) if bias and elementwise_affine else None
		self.weight = tc.ones(*self.normalized_shape) if elementwise_affine else None
		self.bias = tc.zeros(*self.normalized_shape) if bias and elementwise_affine else None
	
	def tg_forward(_, self, x):
		assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
		x = x.layernorm(eps=self.eps, axis=self.axis)
		if not self.elementwise_affine: return x
		return x * self.weight + self.bias
		
class Linear(Module):
	def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
		bound = 1 / math.sqrt(in_features)
		#self.weight = AT(tinygrad.Tensor.uniform(out_features, in_features, low=-bound, high=bound) )
		self.weight = tc.empty(  (out_features, in_features)  )
		internal_init.uniform_(self.weight, a = -bound, b = bound)
		#self.bias = AT( tinygrad.Tensor.uniform(out_features, low=-bound, high=bound) ) if bias else None
		if bias:
			self.bias = tc.empty(  (out_features,)  )
			internal_init.uniform_(self.bias, a = -bound, b = bound)
		else:
			self.bias = None
	
	@property
	def in_features(self):
		return self.weight.shape[1]
	
	@property
	def out_features(self):
		return self.weight.shape[0]
	
	def forward(self, x):
		# disinherit stuff
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		in_features = weight.shape[1]
		out_features = weight.shape[0]
		out_shape = list(x.shape)
		out_shape[-1] = out_features
		out_shape = tuple(out_shape)
		#x = x.reshape(-1, in_features)
		x = x.linear(weight.transpose(), bias)
		#print(x.shape, out_shape)
		#x = x.reshape(out_shape)
		return convert_to_torch(x)
	
	def tg_forward(_, self, x):
		# disinherit stuff
		x, weight, bias = x, self.weight, self.bias
		in_features = weight.shape[1]
		out_features = weight.shape[0]
		out_shape = list(x.shape)
		out_shape[-1] = out_features
		out_shape = tuple(out_shape)
		#x = x.reshape(-1, in_features)
		x = x.linear(weight.transpose(), bias)
		#print(x.shape, out_shape)
		#x = x.reshape(out_shape)
		return x
	
class Embedding(Module):
	def __init__(self,
			vocab_size:int,
			embed_size:int,
			padding_idx = None,
			max_norm = None,
			norm_type = 2.0,
			scale_grad_by_freq = False,
			_weight = None,
			_freeze = False,
			device = None,
			dtype = None
			):
		self.vocab_sz, self.embed_sz = vocab_size, embed_size
		self.weight = tc.empty( (vocab_size, embed_size) )
		internal_init.xavier_uniform_(self.weight )
	
	def tg_forward(parent, self, idx):
		vocab_sz, embed_sz, weight = self.vocab_sz, self.embed_sz, self.weight
		
		original_device = idx.device
		working_device = idx.device
		
		if not tg_device_supports_longlong(weight.device):
			# perform embedding on the CPU as a fallback
			working_device = "CPU"
		
		if not hasattr(parent, 'arange'): parent.arange = tinygrad.Tensor.arange(vocab_sz,
			requires_grad=False, device=working_device, dtype = highest_precision_int(working_device) ).unsqueeze(-1)
		big_shp = idx.shape+(vocab_sz, embed_sz)
		
		
		idx = idx.to(working_device)
		weight = weight.to(working_device)
		
		arange, idx, vals = parent.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), weight.expand(big_shp)
		
		# (-1, 77, 49408, -1)
		inter = (arange == idx)
		
		# (-1, 77, 49408, -1)
		inter2 = inter.mul(vals)
		out = inter2.sum(-2)
		
		return out.to(original_device)
		

class GroupNorm(Module):
	def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
			device=None, dtype=None):
		self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
		self.weight = tc.ones(num_channels) if affine else None
		self.bias = tc.zeros(num_channels) if affine else None
	"""
	def forward(self, x):
		# disinherit stuff
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)
		
		
		
		if weight is None or bias is None: return _cb(x)
		out = x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2))
		return AT(out)
	"""
	def tg_forward(_, self, x):
		# disinherit stuff
		x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)
		if self.weight is None or self.bias is None: return x
		return x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))
		
class MultiheadAttention(Module):
	def __init__(self,
				embed_dim,
				num_heads,
				dropout=0.0,
				bias=True,
				add_bias_kv=False,
				add_zero_attn=False,
				kdim=None, vdim=None,
				batch_first=False,
				device=None,
				dtype=None
			):
		
		if embed_dim <= 0 or num_heads <= 0:
			raise ValueError(
				f"embed_dim and num_heads must be greater than 0,"
				f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
			)
		factory_kwargs = {"device": device, "dtype": dtype}
		super().__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if vdim is not None else embed_dim
		self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

		self.num_heads = num_heads
		self.dropout = dropout
		self.batch_first = batch_first
		self.head_dim = embed_dim // num_heads
		assert (
			self.head_dim * num_heads == self.embed_dim
		), "embed_dim must be divisible by num_heads"

		if not self._qkv_same_embed_dim:
			self.q_proj_weight = Parameter(
				tc.empty((embed_dim, embed_dim), **factory_kwargs)
			)
			self.k_proj_weight = Parameter(
				tc.empty((embed_dim, self.kdim), **factory_kwargs)
			)
			self.v_proj_weight = Parameter(
				tc.empty((embed_dim, self.vdim), **factory_kwargs)
			)
			#self.register_parameter("in_proj_weight", None)
		else:
			self.in_proj_weight = Parameter(
				tc.empty((3 * embed_dim, embed_dim), **factory_kwargs)
			)
			#self.register_parameter("q_proj_weight", None)
			#self.register_parameter("k_proj_weight", None)
			#self.register_parameter("v_proj_weight", None)

		if bias:
			self.in_proj_bias = Parameter(tc.empty(3 * embed_dim, **factory_kwargs))
		self.out_proj = Linear(
			embed_dim, embed_dim, bias=bias, **factory_kwargs
		)

		if add_bias_kv:
			self.bias_k = Parameter(tc.empty((1, 1, embed_dim), **factory_kwargs))
			self.bias_v = Parameter(tc.empty((1, 1, embed_dim), **factory_kwargs))
		else:
			self.bias_k = self.bias_v = None

		self.add_zero_attn = add_zero_attn
		self.max_self_attn_cache_len = 512 # lets just try this lol

	def tg_forward(_, self, q, k, v, key_padding_mask = None,
			need_weights = True, attn_mask = None,
			average_attn_weights = False, is_causal = False):
		if True in (not key_padding_mask is None, average_attn_weights, is_causal):
			# not going to bother with these for now
			raise NotImplementedError
		if hasattr(self, "in_proj_weight"):
			wq, wk, wv = self.in_proj_weight.chunk(3, dim = 0)
		else:
			wq, wk, wv = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
			
		# YAY!!!!!!
		q = q @ wq.T
		k = k @ wk.T
		v = v @ wv.T
		
		if hasattr(self, "in_proj_bias"):
			q = q + self.in_proj_bias[0:self.embed_dim]
			k = k + self.in_proj_bias[self.embed_dim:self.embed_dim*2]
			v = v + self.in_proj_bias[self.embed_dim*2:self.embed_dim*3]
		
		if not self.bias_k is None:
			b += self.bias_k
			v += self.bias_v
		qc = q.chunk(self.num_heads, dim = 1)
		kc = k.chunk(self.num_heads, dim = 1)
		vc = v.chunk(self.num_heads, dim = 1)
		
		assert qc[0].shape[1] == vc[0].shape[1] == kc[0].shape[1] == self.head_dim
		#input(self.out_proj.weight.shape)
		
		att = []
		weights = []
		for head in range(self.num_heads):
			qi, ki, vi = qc[head], kc[head], vc[head]
			hi = tinygrad.Tensor.scaled_dot_product_attention(qi, ki, vi, attn_mask = attn_mask)
			if need_weights:
				#weights.append()
				pass
			att.append(hi)
		weight = tinygrad.Tensor.cat(*att, dim = 1)
		out = self.out_proj(weight)
		#input(out.shape)
		if need_weights:
			# For now, weight just miight be inaccurate :c
			return out, weight[0:out.shape[0], 0:out.shape[0]]
		return (out,)
	
	"""
	def forward(self, query, key, value, key_padding_mask = None,
			need_weights = True, attn_mask = None,
			average_attn_weights = False, is_causal = False):
		if True in (not key_padding_mask is None, average_attn_weights, is_causal):
			# not going to bother with these for now
			raise NotImplementedError
		q, k, v = convert_to_tg(query, key , value)
		if hasattr(self, "in_proj_weight"):
			wq, wk, wv = convert_to_tg(self.in_proj_weight).chunk(3, dim = 0)
		else:
			wq, wk, wv = convert_to_tg(self.q_proj_weight, self.k_proj_weight, self.v_proj_weight)
			
		# YAY!!!!!!
		q = q @ wq.T
		k = k @ wk.T
		v = v @ wv.T
		
		if hasattr(self, "in_proj_bias"):
			inb = convert_to_tg(self.in_proj_bias)
			q = q + inb[0:self.embed_dim]
			k = k + inb[self.embed_dim:self.embed_dim*2]
			v = v + inb[self.embed_dim*2:self.embed_dim*3]
		
		if not self.bias_k is None:
			bk, bv = convert_to_tg(bias_k, bias_v)
			b += bk
			v += bv
		qc = q.chunk(self.num_heads, dim = 1)
		kc = k.chunk(self.num_heads, dim = 1)
		vc = v.chunk(self.num_heads, dim = 1)
		
		assert qc[0].shape[1] == vc[0].shape[1] == kc[0].shape[1] == self.head_dim
		#input(self.out_proj.weight.shape)
		
		att = []
		weights = []
		for head in range(self.num_heads):
			qi, ki, vi = qc[head], kc[head], vc[head]
			hi = tinygrad.Tensor.scaled_dot_product_attention(qi, ki, vi, attn_mask = attn_mask)
			if need_weights:
				#weights.append()
				pass
			att.append(hi)
		weight = convert_to_torch( tinygrad.Tensor.cat(*att, dim = 1) )
		out = self.out_proj(weight)
		#input(out.shape)
		if need_weights:
			# For now, weight just miight be inaccurate :c
			return out, weight[0:out.shape[0], 0:out.shape[0]]
		return (out,)
	"""

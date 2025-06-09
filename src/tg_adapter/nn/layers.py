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
	
	def forward(self, inp):
		inp = convert_to_tg(inp)
		out = inp.avg_pool2d(self._kernel_size, self._stride,
			1, self._padding, self._ceil_mode, self._count_include_pad)
		return AT(out)

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
	
	def forward(self, inp):
		return AT(inp.tg.dropout(self._p) )

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
	
	def forward(self, x):
		x, weight, bias = x.tg, self.weight.tg, self.bias.tg
		x = x.conv2d(weight, bias, self.groups, self.stride, self.dilation, self.padding)
		return AT(x)

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
	
	def forward(self, x):
		x, weight, bias, = convert_to_tg(x, weight, bias)
		x = x.conv_transpose2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding, self.output_padding)
		return AT(x)

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
	
	def forward(self, x):
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
		x = x.layernorm(eps=self.eps, axis=self.axis)
		if not self.elementwise_affine: return AT(x)
		return AT(x * weight + bias)
		
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
		
class Embedding(Module):
	def __init__(self, vocab_size:int, embed_size:int):
		self.vocab_sz, self.embed_sz = vocab_size, embed_size
		self.weight = tc.empty( (vocab_size, embed_size) )
		internal_init.xavier_uniform_(self.weight )
	
	def forward(self, idx):
		vocab_sz, embed_sz, weight, idx = convert_to_tg(self.vocab_sz, self.embed_sz, self.weight, idx)
		
		original_device = idx.device
		working_device = idx.device
		
		if not tg_device_supports_longlong(weight.device):
			# perform embedding on the CPU as a fallback
			working_device = "CPU"
		
		if not hasattr(self, 'arange'): self.arange = tinygrad.Tensor.arange(vocab_sz,
			requires_grad=False, device=working_device, dtype = highest_precision_int(working_device) ).unsqueeze(-1)
		big_shp = idx.shape+(vocab_sz, embed_sz)
		
		
		idx = idx.to(working_device)
		weight = weight.to(working_device)
		
		arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), weight.expand(big_shp)
		
		# (-1, 77, 49408, -1)
		inter = (arange == idx).realize()
		
		# (-1, 77, 49408, -1)
		inter2 = inter.mul(vals).realize()
		out = inter2.sum(-2).realize()
		
		out = out.to(original_device)
		return AT(out)
		

class GroupNorm(Module):
	def __init__(self, num_groups, num_channels, eps=1e-05, affine=True,
			device=None, dtype=None):
		self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
		self.weight = tc.ones(num_channels) if affine else None
		self.bias = tc.zeros(num_channels) if affine else None
		#self.weight = AT(tinygrad.Tensor.ones(num_channels) ) if affine else None
		#self.bias = AT( tinygrad.Tensor.zeros(num_channels) ) if affine else None
	
	def forward(self, x):
		# disinherit stuff
		x, weight, bias = convert_to_tg(x, self.weight, self.bias)
		x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)
		
		
		
		if weight is None or bias is None: return _cb(x)
		out = x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2))
		return AT(out)

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

		

	def forward(self, query, key, value, **kwargs):
		if self._qkv_same_embed_dim:
			# still have to figure out how to split everything correctly
			raise NotImplementedError
		query, key, value = convert_to_tg(query, key, value)
		for head in self.num_heads:
			input(head)
		raise NotImplementedError
		scaled_dot_product_attention()
		
"""
class MultiHeadAttention:
	def __init__(self, n_state, n_head, kv_caching: Literal['cross', 'self']=None, max_self_attn_cache_len=None):
		self.n_head = n_head
		self.query = nn.Linear(n_state, n_state)
		self.key = nn.Linear(n_state, n_state, bias=False)
		self.value = nn.Linear(n_state, n_state)
		self.out = nn.Linear(n_state, n_state)

		self.kv_caching = kv_caching
		self.max_self_attn_cache_len = max_self_attn_cache_len

  def __call__(self, x:Tensor, xa:Optional[Tensor]=None, mask:Optional[Tensor]=None, len: Union[Variable,int]=None):
    if self.kv_caching == 'cross':
      if xa is not None:
        k, v = self.key(xa), self.value(xa)
        if not hasattr(self, 'cache_k'):
          self.cache_k, self.cache_v = k, v
        else:
          self.cache_k.assign(k).realize()
          self.cache_v.assign(v).realize()
      else:
        k, v = self.cache_k, self.cache_v
    else:
      k, v = self.key(x), self.value(x)
      if self.kv_caching == 'self':
        if not hasattr(self, 'cache_k'):
          self.cache_k = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
          self.cache_v = Tensor.zeros(x.shape[0], self.max_self_attn_cache_len, x.shape[2])
        k = self.cache_k.shrink((None, (0, len), None)).cat(k, dim=1)
        v = self.cache_v.shrink((None, (0, len), None)).cat(v, dim=1)
        padding = self.max_self_attn_cache_len-len-x.shape[1]
        self.cache_k.assign(k.pad((None, (0, padding), None)).contiguous()).realize()
        self.cache_v.assign(v.pad((None, (0, padding), None)).contiguous()).realize()

    q = self.query(x)
    n_ctx = q.shape[1]
    assert(q.shape[-1] == k.shape[-1] == v.shape[-1])
    head_dim = q.shape[-1] // self.n_head
    q = q.reshape(*q.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    k = k.reshape(*k.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(*v.shape[:2], self.n_head, head_dim).permute(0, 2, 1, 3)
    attn = Tensor.scaled_dot_product_attention(q, k, v, mask[:n_ctx,:n_ctx] if mask is not None else None)
    wv = attn.permute(0, 2, 1, 3).flatten(start_dim=2)
    return self.out(wv)
"""

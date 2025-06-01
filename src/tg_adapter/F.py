import tinygrad
import numpy as np
from .tensor import convert_to_tg, convert_to_torch, assert_same_device
from .debugging import maybe_realize
from copy import deepcopy
import math

def interpolate(inp,
		size=None,
		scale_factor=None,
		mode='nearest',
		align_corners=None,
		recompute_scale_factor=None,
		antialias=False
		):
	if align_corners is None:
		align_corners = False
	if recompute_scale_factor is True:
		# not dealing with this crap for now
		raise NotImplementedError
	if antialias:
		# or this crap either lol
		raise NotImplementedError
	
	size = list(inp.shape)
	len_size = len(size)
	if not scale_factor is None:
		if isinstance(scale_factor, tuple):
			assert len(scale_factor) == len(size) - 2
			for i, sf in enumerate(scale_factor):
				size[i+2] = int(size[i+2]*scale_factor)
		else:
			for i in range(len_size - 2):
				size[i+2] = int(size[i+2] * scale_factor)
		size = tuple(size)
	else:
		assert isinstance(size, tuple)
	return convert_to_torch( inp.interpolate(size, mode, align_corners) )
	
def group_norm(x, num_groups, weight = None, bias = None, eps = 1.0e-5):
	# derived from the tinygrad source code c:
	x, weight, bias = convert_to_tg(x, weight, bias)
	assert_same_device(x.device, weight, bias)
	x = x.reshape(x.shape[0], num_groups, -1).layernorm(eps=eps).reshape(x.shape)

	if weight is None or bias is None: return x
	# elementwise_affine on channels
	return convert_to_torch(x * weight.reshape(1, -1, *[1] * (x.ndim-2)) + bias.reshape(1, -1, *[1] * (x.ndim-2)) )
	
def scaled_dot_product_attention(query, key, value, attn_mask=None,
		dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
	query, key, value, attn_mask = convert_to_tg( (query, key, value, attn_mask) )
	assert_same_device(query.device, key, value, attn_mask)
	if enable_gqa:
		# not ever sure what this is
		raise NotImplementedError
	if not scale is None:
		# divide the custom scale factor by the internal scale factor
		internal_scale = 1.0 / np.sqrt(query.size(-1))
		scale_factor = scale/internal_scale
		
		# then multiply the query by it
		query = query * scale_factor
	return convert_to_torch( query.scaled_dot_product_attention(key, value, attn_mask,
		dropout_p, is_causal) )

def pad(inp, pad, mode='constant', value=None):
	if value is None:
		value = 0.0
	if mode != "constant":
		raise NotImplementedError
	inp = convert_to_tg(inp)
	return convert_to_torch(inp.pad(pad, mode, value) )
	
def embedding(inp, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
	raise NotImplementedError


def _true_gelu(x: tinygrad.Tensor):
	return x*(1 + (x / math.sqrt(2) ).erf() ) / 2

def gelu(x, approximation = None):
	x = convert_to_tg(x)
	if approximation is None:
		# use my own implementation, _true_gelu
		return convert_to_torch(_true_gelu(x) )
	elif approximation == "tanh":
		# Tinygrad uses the tanh approximation internally
		return convert_to_torch(x.gelu() )
	else:
		raise ValueError

mish = lambda x: convert_to_torch(convert_to_tg(x).mish() )
sigmoid = lambda x: convert_to_torch(convert_to_tg(x).sigmoid() )

def cumprod(inp, dim, dtype=None, out=None):
	# first, get the slices used in the __getitem__ call for each element
	inp = convert_to_tg(inp)
	slices = []
	for i in range(len(inp.shape)):
		slices.append(slice(None, None, None) )
	
	outputs = []
	for i in range(inp.shape[dim] ):
		slices[dim] = slice(0, i + 1, None)
		new_shape = list(inp.shape)
		new_shape[dim] = -1
		new_shape = tuple(new_shape)
		outputs.append(inp[slices].prod(dim).reshape(new_shape) )
	outputs = convert_to_torch(outputs)
	return cat(outputs, dim)
	
# easier than rearranging huggingface code lol
def chunk(inp, chunks: int, dim: int = 0):
	inp = convert_to_tg(inp)
	return convert_to_torch(inp.chunk(chunks, dim) )
	
def clamp(inp, min = None, max = None):
	inp = convert_to_tg(inp)
	return convert_to_torch(inp.clamp(min, max) )
	
def cat(tensors, dim = 0):
	tbase = tensors[0].tg
	trest = convert_to_tg( tuple(tensors[1:]) )
	assert_same_device(tbase.device, trest)
	return convert_to_torch(tbase.cat(*trest, dim = dim) )
	
def normalize(inp, p = 2.0, dim = 1, eps = 1.0e-12, out = None):
	raise NotImplementedError

import torch
import tg_adapter
from testing_utils import *

def test_cumprod():
	from tg_adapter import F as tinyF
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	test_function( (inp, 0), {}, torch.cumprod, tinyF.cumprod)


def test_cat():
	a = make_test_data(40, 2, 5)
	b = make_test_data(40, 2, 5)
	for i in range(3):
		test_function( ([a, b], i), {}, torch.cat, tg_adapter.cat)
	test_function( ([a, b], -1), {}, torch.cat, tg_adapter.cat)

def test_interpolate():
	shape = (2, 3, 6, 6)
	a = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
	args = (a, None, 2.0)
	test_function(args, {}, torch_function = torch.nn.functional.interpolate, tinygrad_function = tg_adapter.F.interpolate)

def _test_unary(torch_function, tinygrad_function, data = None):
	if data is None:
		shape = (4, 2, 6, 8)
		data = make_test_data(*shape)
	test_function( (data), {}, torch_function, tinygrad_function)
	

def test_scaled_dot_product_attention():
	q = make_test_data(1, 8, 4096, 40)
	k = make_test_data(1, 8, 4096, 40)
	v = make_test_data(1, 8, 4096, 40)
	test_function( [q, k, v], {}, torch.nn.functional.scaled_dot_product_attention, tg_adapter.F.scaled_dot_product_attention)

def test_gelu():
	_test_unary(torch.nn.functional.gelu, tg_adapter.F.gelu, np.arange(1000).astype(np.float32) - 500.0 )

def test_sigmoid():
	_test_unary(torch.nn.functional.sigmoid, tg_adapter.F.sigmoid, np.arange(1000).astype(np.float32) - 500.0 )

def test_mish():
	_test_unary(torch.nn.functional.mish, tg_adapter.F.mish, np.arange(1000).astype(np.float32) - 500.0 )


def _test_chunk(dim):	
	data = make_test_data(16, 8, 4, 8)
	test_function([data, 2, dim], {}, torch.chunk, tg_adapter.chunk)

def test_chunk():
	for i in range(4):
		_test_chunk(i)
	_test_chunk(-1)

def test_clamp():
	data = make_test_data(3, 5, 8)
	test_function([data, 0.0, 0.5], {}, torch.clamp, tg_adapter.F.clamp)

def test_stack():
	tensors = []
	for i in range(3):
		tensors.append(make_test_data(4, 4, 4) )
	
	for i in range(3):
		test_function( [tensors, i], {}, torch.stack, tg_adapter.stack )

def test_pow():
	x = np.abs(make_test_data(3, 4, 5) )
	y = make_test_data(3, 4, 5)
	test_function([x, y], {}, torch.pow, tg_adapter.pow)
	
def test_magic_pow():
	a = np.abs(make_test_data(3, 4, 7) )
	def pow_impl(x, y):
		return x ** y
	test_function([a, 0.5], {}, pow_impl, pow_impl)

def test_max():
	a = make_test_data(3, 2, 5, 8)
	test_function([a], {}, torch.max, tg_adapter.max)
	
def test_min():
	a = make_test_data(3, 2, 5, 8)
	test_function([a], {}, torch.min, tg_adapter.min)
	
def test_radd():
	raise NotImplementedError
	
def test_rsub():
	a = np.abs(make_test_data(3, 4, 7) )
	def _rsub_test(a, b):
		return a - b
	test_function([1, a], {}, _rsub_test, _rsub_test)

def test_all_operators():
	test_rsub()
	test_chunk()
	test_cumprod()
	test_cat()
	test_interpolate()
	test_scaled_dot_product_attention()
	
	test_gelu()
	test_sigmoid()
	test_mish()
	
	test_clamp()
	test_stack()
	test_pow()
	test_magic_pow()
	test_max()
	test_min()
	

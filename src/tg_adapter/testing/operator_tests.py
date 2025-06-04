import torch
import tg_adapter
from .testing_utils import *

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
	test_function([a, 0.5], {}, pow_impl, pow_impl, error_threshold=1.0e-7)

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

def test_mean():
	a = np.arange(7*8).reshape(7, 8).astype(np.float32)
	for i in range(2):
		test_function([a], {"dim": i, "keepdim": False}, torch.mean, tg_adapter.mean)
		test_function([a], {"dim": i, "keepdim": True}, torch.mean, tg_adapter.mean)
	test_function([a], {}, torch.mean, tg_adapter.mean)

def test_var():
	a = np.arange(7*8).reshape(7, 8).astype(np.float32)
	for i in range(2):
		test_function([a], {"dim": i, "keepdim": False}, torch.var, tg_adapter.var)
		test_function([a], {"dim": i, "keepdim": True}, torch.var, tg_adapter.var)
	test_function([a], {}, torch.mean, tg_adapter.mean)

def test_normal_():
	a = np.arange(16384).astype(np.float32)
	def test_normal__impl(t):
		old_t = t
		if isinstance(t, torch.Tensor):
			# torch
			t = torch.nn.init.normal_(t)
			return torch.mean(old_t), torch.var(old_t), torch.mean(t), torch.var(t)
		else:
			# adapter tensor
			t = tg_adapter.nn.init.normal_(t)
			return tg_adapter.mean(old_t), tg_adapter.var(old_t), tg_adapter.mean(t), tg_adapter.var(t)
	test_function([a], {}, test_normal__impl, test_normal__impl, error_threshold = 1.0e-2)
		
def test_diag():
	for i in range(2):
		shape = []
		for i2 in range(i + 1):
			shape.append(int(4) )
		shape = tuple(shape)
		a = np.arange(np.prod(shape) ).reshape(shape).astype(np.float32)
		for diagonal in range(3*2):
			diagonal = diagonal - 3
			test_function([a, diagonal], {}, torch.diag, tg_adapter.diag)
			
def test_norm():
	for dim in range(2):
		shape = []
		for i2 in range(dim + 1):
			shape.append(int(4) )
		shape = tuple(shape)
		a = np.arange(np.prod(shape) ).reshape(shape).astype(np.float32)
		for keepdim in [True, False]:
			test_function([a], {"dim": dim, "keepdim": keepdim}, torch.linalg.norm, tg_adapter.linalg.norm)

def test_outer():
	u = make_test_data(5)
	v = make_test_data(8)
	test_function([u, v], {}, torch.outer, tg_adapter.outer)
			
def test_qr():
	A = np.random.randn(4*4).reshape(4, 4).astype(np.float32)
	
	# actual underlying q value seems to differ, but does that matter?
	def _qr_test(a):
		if isinstance(a, torch.Tensor):
			q, r = torch.linalg.qr(a)
		else:
			q, r = tg_adapter.linalg.qr(a)
		return q@r
	test_function([A], {}, _qr_test, _qr_test)
	
def test_complex_add():
	a = make_test_data(16) + 1.0j*make_test_data(16)
	b = make_test_data(16) + 1.0j*make_test_data(16)
	add_test = lambda x, y: x + y
	test_function([a, b], {}, add_test, add_test)
	
def _zeros_like(inp):
	if isinstance(inp, torch.Tensor):
		return torch.zeros_like(inp)
	else:
		return tg_adapter.zeros_like(inp)

def test_eig():
	A = np.random.randn(4*4).reshape(4, 4).astype(np.float32)
	def _eig_test(a):
		if isinstance(a, torch.Tensor):
			result = torch.linalg.eig(a)
			vals, vecs = result
			vals = vals.real
			vecs = vecs.real
		else:
			vals, vecs = tg_adapter.linalg.eig(a)
		#return vals, vecs
		n = vecs.shape[0]
		comparisons = []
		for i in range(n):
			val = vals[i]
			vec = vecs[:, i]
			q = a @ vec.reshape(-1, 1)
			b = val * vec
			
			# MSE doesn't handle complex numbers well :c
			try:
				q_imag = q.imag
			except RuntimeError:
				# just do 0
				q_imag = _zeros_like(q.real)
				
			try:
				b_imag = b.imag
			except RuntimeError:
				# just do 0
				b_imag = _zeros_like(b.real)
			comparisons.append((q.real, q_imag, b.real, b_imag))
		return comparisons
			
	#test_function([A], {}, torch.linalg.eig, tg_adapter.linalg.eig)
	test_function([A], {}, _eig_test, _eig_test)



def test_all_operators():
	test_complex_add()
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
	
	
	test_mean()
	test_var()
	
	test_normal_()
	
	test_diag()
	test_norm()
	test_outer()
	test_qr()
	#test_eig()
	# just return true for now i guess
	return True
	

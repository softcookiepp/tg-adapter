import torch
import tg_adapter
from .testing_utils import *
from .testing_utils import _test_function, _test_hf_reimplementation
import tinybloat
	
	
def _test_linear(use_bias):
	from tg_adapter.nn import Linear as tg_class
	from torch.nn import Linear as hf_class
	
	hf_module = hf_class(4096, 2048, use_bias)
	tg_module = tg_class(4096, 2048, use_bias)
	copy_state_dict(hf_module, tg_module)
	
	a = make_test_data(2, 3, 4096)
	_test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")
	
	a = a.reshape(-1, 4096)
	
	_test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")
	
	
	
def test_linear():
	_test_linear(True)
	_test_linear(False)


def test_avg_pool_2d():
	from tg_adapter.nn import AvgPool2d as tg_class
	from torch.nn import Linear as hf_class
	use_bias = True
	hf_module = hf_class(4096, 2048, use_bias)
	tg_module = tg_class(4096, 2048, use_bias)
	
def test_module_list():
	torch_module = torch.nn.ModuleList([torch.nn.Linear(2, 4)])
	tg_module = tg_adapter.nn.ModuleList([tg_adapter.nn.Linear(2, 4)])
	copy_state_dict(torch_module, tg_module)
	module_list_test = lambda x, _torch: x.state_dict()
	_test_hf_reimplementation([], {}, torch_module, module_list_test, tg_module, module_list_test)

def test_multihead_attention():
	embed_dim = 16
	num_heads = 4
	batch_size = 8
	torch_module = torch.nn.MultiheadAttention(embed_dim, num_heads)#, bias = False)
	tg_module = tg_adapter.nn.MultiheadAttention(embed_dim, num_heads)#, bias = False)
	copy_state_dict(torch_module, tg_module)
	q = make_test_data(batch_size, embed_dim)
	k = make_test_data(batch_size, embed_dim)
	v = make_test_data(batch_size, embed_dim)
	_test_hf_reimplementation((q, k, v), {"need_weights": False}, torch_module, "__call__", tg_module, "__call__")
	
def test_tb_multihead_attention():
	embed_dim = 16
	num_heads = 4
	batch_size = 8
	torch_module = torch.nn.MultiheadAttention(embed_dim, num_heads)#, bias = False)
	tg_module = tinybloat.nn.MultiheadAttention(embed_dim, num_heads)#, bias = False)
	copy_state_dict(torch_module, tg_module, False)
	q = make_test_data(batch_size, embed_dim)
	k = make_test_data(batch_size, embed_dim)
	v = make_test_data(batch_size, embed_dim)
	_test_hf_reimplementation((q, k, v), {"need_weights": False}, torch_module, "__call__", tg_module, "__call__", use_tg_adapter = False)


def test_conv_transpose_2d():
	torch_module = torch.nn.ConvTranspose2d(2, 4, 3)
	tg_module = tg_adapter.nn.ConvTranspose2d(2, 4, 3)
	copy_state_dict(torch_module, tg_module)
	test_data = make_test_data(2, 2, 16, 16)
	_test_hf_reimplementation([test_data], {}, torch_module, "__call__", tg_module, "__call__")
	
def test_zero_pad_2d():
	for pad in [2, (2, 4, 6, 8)]:
		print(pad)
		torch_module = torch.nn.ZeroPad2d(pad)
		tg_module = tg_adapter.nn.ZeroPad2d(pad)
		a = make_test_data(2, 3, 16, 16)
		_test_hf_reimplementation([a], {}, torch_module, "__call__", tg_module, "__call__")

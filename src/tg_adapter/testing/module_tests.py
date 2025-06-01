import torch
import tg_adapter
from testing_utils import *


def _test_geglu(use_bias = True):
	from tiny_hf.diffusers.models.activations import GEGLU as tg_class
	from diffusers.models.activations import GEGLU as hf_class
	
	hf_module = hf_class(4096, 2048, use_bias)
	tg_module = tg_class(4096, 2048, use_bias)
	copy_state_dict(hf_module, tg_module)
	a = make_test_data(1, 4096)
	test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")

def test_geglu():
	_test_geglu(True)
	_test_geglu(False)
	
	
def _test_linear(use_bias):
	from tg_adapter.nn import Linear as tg_class
	from torch.nn import Linear as hf_class
	
	hf_module = hf_class(4096, 2048, use_bias)
	tg_module = tg_class(4096, 2048, use_bias)
	copy_state_dict(hf_module, tg_module)
	
	a = make_test_data(2, 3, 4096)
	test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")
	
	a = a.reshape(-1, 4096)
	
	test_hf_reimplementation([a], {}, hf_module, "__call__", tg_module, "__call__")
	
	
	
def test_linear():
	_test_linear(True)
	_test_linear(False)


def test_avg_pool_2d():
	from tg_adapter.nn import AvgPool2d as tg_class
	from torch.nn import Linear as hf_class
	hf_module = hf_class(4096, 2048, use_bias)
	tg_module = tg_class(4096, 2048, use_bias)

def test_modules():
	test_geglu()
	test_linear()

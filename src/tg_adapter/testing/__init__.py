try:
	import torch
except ImportError:
	raise ImportError("The testing suite for tg_adapter requires pytorch to be installed for use as a reference.\nA CPU-only version can be installed by running 'install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'")

from .operator_tests import test_all_operators
from .module_tests import test_modules
import pytest

def run_tests():
	pytest.main(["--pyargs", "tg_adapter.testing.peepee"])
	result = True
	#result = test_all_operators()
	#test_modules()
	if result:
		print("All tests passing!")
	else:
		print("Some tests failed, please examine output")

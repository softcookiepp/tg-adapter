try:
	import torch
except ImportError:
	raise ImportError("The testing suite for tg_adapter requires pytorch to be installed for use as a reference.\nA CPU-only version can be installed by running 'install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'")

import pytest

def run_tests():
	pytest.main(["--pyargs", "tg_adapter.testing.operator_tests", "tg_adapter.testing.module_tests"])

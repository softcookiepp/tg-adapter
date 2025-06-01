try:
	import torch
except ImportError:
	raise ImportError("The testing suite for tg_adapter requires pytorch to be installed for use as a reference.\nA CPU-only version can be installed by running 'install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu'")

def run_tests():
	raise NotImplementedError

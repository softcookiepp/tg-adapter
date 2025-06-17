import tinygrad
from .file_pointer import FilePointer
from ..tensor import AdapterTensor
from . import torch

def safe_open(fn: str, framework = "pt", device = None):
	if not framework in ["pt"]:
		raise NotImplementedError
	if not device in [None]:
		raise NotImplementedError
	

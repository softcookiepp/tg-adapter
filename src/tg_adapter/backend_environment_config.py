# Allows user override of which backends torch.Tensor.cuda(), torch.Tensor.cuda(), etc. point to

import os

_ENVIRON = {
	"TGA_ALL_ACCELERATOR_OVERRIDE" : None, # just for now hehe
	"TGA_CUDA_OVERRIDE" : "GPU",
	"TGA_CPU_OVERRIDE" : "CPU",
	"TGA_MPS_OVERRIDE" : "METAL",
	"TGA_XPU_OVERRIDE" : "XPU"
}

for key in _ENVIRON.keys():
	if key in os.environ.keys():
		_ENVIRON[key] = os.environ[key]

# torch backends to tinygrad backends
_BACKEND_TABLE = {
	"cpu" : "CPU",
	"cuda" : _ENVIRON["TGA_CUDA_OVERRIDE"],
	"xpu" : _ENVIRON["TGA_XPU_OVERRIDE"],
	"mps" : _ENVIRON["TGA_MPS_OVERRIDE"]
}

def torch_dev_to_tiny(torch_dev, idx = None):
	if torch_dev is None:
		return None
	if not isinstance(torch_dev, str):
		# device object
		idx = torch_dev.idx
		torch_dev = torch_dev.name
	if idx is None:
		return _BACKEND_TABLE[torch_dev]
	else:
		return f"{_BACKEND_TABLE[torch_dev]}:{idx}"
	

def tiny_dev_to_torch(tiny_dev):
	name = tiny_dev
	idx = None
	if ":" in tiny_dev:
		name, idx = tuple(tiny_dev.split(":") )
		idx = int(idx)
	for k, v in _BACKEND_TABLE.items():
		if name == v:
			if idx is None:
				return k
			else:
				return f"{k}:{idx}"
	raise ValueError

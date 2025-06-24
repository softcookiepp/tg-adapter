import tinygrad
import pickle
from .tensor import AdapterTensor

def save(*args, **kwargs):
	raise NotImplementedError

def load(f, map_location=None, pickle_module=pickle, *, weights_only=True,
		mmap=None, **pickle_load_args):
	# TODO: how do i load a torch model into tinygrad again?
	# i forget
	
	if not weights_only:
		raise NotImplementedError("Can't implement reconstruction of entire torch classes, sorry :c")
	
	# has to be str for now :c
	assert type(f) in [str]
	
	# So I believe we can do tinygrad.nn.state.torch_load...
	state_dict = tinygrad.nn.state.torch_load(f)
	new_state_dict = {}
	for k, v in state_dict.items():
		new_state_dict[k] = AdapterTensor( v.to("CPU") )
	return new_state_dict

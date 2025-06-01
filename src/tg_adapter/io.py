import tinygrad
import pickle

def save(*args, **kwargs):
	raise NotImplementedError

def load(f, map_location=None, pickle_module=pickle, *, weights_only=True,
		mmap=None, **pickle_load_args):
	# TODO: how do i load a torch model into tinygrad again?
	# i forget
	raise NotImplementedError
	

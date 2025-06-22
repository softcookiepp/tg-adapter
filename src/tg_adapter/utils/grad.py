import tinygrad

def dummy(*args):
	if len(args) == 1:
		return args[0]
	return args

def no_grad(orig_func = None):
	# this does literally nothing
	return dummy
	#return tinygrad.Tensor.test()

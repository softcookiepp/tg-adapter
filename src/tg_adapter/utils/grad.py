import tinygrad

def no_grad(orig_func = None):
	# this does literally nothing
	return orig_func
	#return tinygrad.Tensor.test()

# Some functions in vanilla tinygrad seem to generate problematic kernels
# that sometimes do not compile.
# Therefore, some will be reimplemented here.
import tinygrad

def max(inp, dim = None, axis = None, keepdim = False):
	if dim is None:
		dim = axis
	if dim is None:
		# tensor needs to be flattened
		inp = inp.reshape(-1)
		dim = 0
	
	# The strategy will be to split inp into chunks,
	# get the argmax of the given chunk, then
	# do some goodie stuffs that i kinda forget
	
	# So, we need to get the chunk count.
	chunk_count = inp.shape[dim] - 1
	while chunk_count > 0:
		if inp.shape[dim] % chunk_count == 0:
			break
		chunk_count -= 1
	if chunk_count == 0:
		raise ValueError
	chunk_size = inp.shape[dim] // chunk_count
	chunks = inp.chunk(chunk_count, dim = dim)
	argmax_chunks = []
	for i, chunk in enumerate(chunks):
		idx = chunk.contiguous().argmax(dim)

def argmax(inp, dim = None, axis = None, keepdim = False):
	if dim is None:
		dim = axis
	if dim is None:
		# tensor needs to be flattened
		inp = inp.reshape(-1)
		dim = 0
	return (inp == max(inp, dim, axis, keepdim) ).sum()

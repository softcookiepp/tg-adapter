# Some functions in vanilla tinygrad seem to generate problematic kernels
# that sometimes do not compile.
# Therefore, some will be reimplemented here.
import tinygrad

def argmax(inp, dim = None, axis = None, keepdim = False):
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
	chunks = inp.chunk(chunk_count, dim = dim)
	for chunk in chunks:
		print(chunk.contiguous().argmax(dim).realize().numpy() )
	input("pls work lul")
	raise NotImplementedError

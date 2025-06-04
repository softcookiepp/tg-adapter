import tinygrad
import numpy as np

def _slice_to_square(t, offset = 0):
	if len(t.shape) == 1:
		return t
	n = min(t.shape)
	if offset >= 0:
		return t[0:n - offset, offset:]
	else:
		return t[0 - offset:, 0: offset]

def diag(t, *args, **kwargs):
	offset = 0
	if "diagonal" in kwargs.keys():
		offset = kwargs["diagonal"]
	elif len(args) > 0:
		offset = args[0]
	t_original = t
	t = _slice_to_square(t, offset)
	e = tinygrad.Tensor.eye(t.shape[0], dtype = t.dtype, device = t.device)
	if len(t.shape) == 1:
		# make diagonal matrix from 1-D tensor
		out = t.expand( (t.shape[0], t.shape[0]) ) * e
		if offset < 0:
			out = out.pad( (0, abs(offset), abs(offset), 0) )
		elif offset > 0:
			# pad
			out = out.pad( (offset, 0, 0, offset) )
		return out
	elif len(t.shape) == 2:
		# make 1-D array from 2-D tensor
		out = (t*e).sum(0, keepdim = False).squeeze()
		if len(out.shape) == 0:
			out = out.unsqueeze(-1)
		return out
	else:
		raise RuntimeError(f"Expected 2D or 1D tensor, but got {len(t.shape) }D instead.")

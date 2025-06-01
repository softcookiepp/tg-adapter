REALIZE_ASAP = True
KEEP_INPUT_TENSORS = False

def maybe_realize(t):
	if REALIZE_ASAP:
		if hasattr(t, "realize"):
			return t.realize()
		elif hasattr(t, "tg"):
			t.tg.realize()
			return t
		raise ValueError
	return t

class InputSpec:
	def __init__(self, *args, **kwargs):
		# Class for storing the input specification of functions for debugging purposes
		self._args = args
		self._kwargs = kwargs
		raise NotImplementedError

import os
import tinybloat

REALIZE_ASAP = False
REALIZE_MODULE_OUTPUT = False
REALIZE_MODULE_DEPTH = 0
KEEP_INPUT_TENSORS = False

if "TGA_REALIZE_ASAP" in os.environ.keys():
	REALIZE_ASAP = bool(int(os.environ["TGA_REALIZE_ASAP"]))
if "TGA_REALIZE_MODULE_OUTPUT" in os.environ.keys():
	REALIZE_MODULE_OUTPUT = bool(int(os.environ["TGA_REALIZE_MODULE_OUTPUT"] ) )
elif "TGA_REALIZE_MODULE_DEPTH" in os.environ.keys():
	REALIZE_MODULE_DEPTH = int(os.environ["TGA_REALIZE_MODULE_DEPTH"])

if "TGA_KEEP_INPUTS" in os.environ.keys():
	KEEP_INPUT_TENSORS = bool(int(os.environment["TGA_KEEP_INPUTS"]) )

def _realize(t):
	if hasattr(t, "realize"):
		return t.realize()
	elif hasattr(t, "tg"):
		t.tg.realize()
		return t
	raise ValueError

def maybe_realize(t):
	if REALIZE_ASAP:
		return _realize(t)
	return t
	

def realize_module_status():
	return REALIZE_MODULE_OUTPUT

def get_realize_depth():
	return REALIZE_MODULE_DEPTH

class InputSpec:
	def __init__(self, *args, **kwargs):
		# Class for storing the input specification of functions for debugging purposes
		self._args = args
		self._kwargs = kwargs
		raise NotImplementedError

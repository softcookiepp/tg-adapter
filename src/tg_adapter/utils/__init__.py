from . import checkpoint
from .tg_helpers import *
from .grad import no_grad
from . import hooks
from . import _pytree

def process_shape(*args):
	if isinstance(args[0], tuple):
		return args[0]
	return args

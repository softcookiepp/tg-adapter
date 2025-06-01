from .module import Module
from .layers import *
from . import init
from .losses import *
from .module_list import ModuleList

# lets see if this runs lmao
#from tinygrad.nn import *
# on second thought, this is a really really bad idea
# a lot of interoperability is lost if we do this :c
# soo we have to use the tinygrad functional api to re-create conv and other layers

from .activations import *
from . import utils


# functional
from .. import F as functional

# aliases for torch classes/functions
from .aliases import *

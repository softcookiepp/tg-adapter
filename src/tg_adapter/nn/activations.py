from .module import Module
from ..tensor import AdapterTensor as AT

class ActivationLambda(Module):
	def __init__(self, act_fn, inplace = False):
		if inplace:
			raise NotImplementedError
		self._act_fn = act_fn
	
	def forward(self, x):
		return AT( self._act_fn(x.tg ) )

def SiLU(inplace = False):
	return ActivationLambda(lambda x: x.silu(), inplace)

def Mish(inplace = False):
	return ActivationLambda(lambda x: x.mish(), inplace)

def GELU(inplace = False):
	return ActivationLambda(lambda x: x.gelu(), inplace)

def ReLU(inplace = False):
	return ActivationLambda(lambda x: x.relu(), inplace)

def LeakyReLU(negative_slope = 0.01, inplace = False):
	return ActivationLambda(lambda x: x.leaky_relu(negative_slope), inplace)

def ReLU6(inplace = False):
	return ActivationLambda(lambda x: x.relu6(), inplace)

def Sigmoid(inplace = False):
	return ActivationLambda(lambda x: x.sigmoid(), inplace)
	
def Tanh(inplace = False):
	return ActivationLambda(lambda x: x.tanh(), inplace)

def PReLU(inplace = False):
	return ActivationLambda(lambda x: x.pelu(), inplace)

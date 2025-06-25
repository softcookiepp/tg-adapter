from .module import Module

class CrossEntropyLoss(Module):
	def __init__(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', label_smoothing=0.0):
		super().__init__()
		raise NotImplementedError
	
	def forward(self, a, b):
		raise NotImplementedError

class BCEWithLogitsLoss(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError

class MSELoss(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError

class L1Loss(Module):
	def __init__(self, *args, **kwargs):
		raise NotImplementedError


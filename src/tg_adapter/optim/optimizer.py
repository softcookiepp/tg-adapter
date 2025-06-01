import tinygrad


class Optimizer:
	def __init__(self, *args, **kwargs):
		raise NotImplementedError
	def step(self):
		raise NotImplementedError

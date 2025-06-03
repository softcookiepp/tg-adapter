
class linalg_eig:
	def __init__(self, eigenvalues, eigenvectors):
		self._values = eigenvalues
		self._vectors = eigenvectors
	
	@property
	def eigenvalues(self):
		return self._values
	
	@property
	def eigenvectors(self):
		return self._vectors
	
	def __iter__(self):
		return iter( (self._values, self._vectors) )

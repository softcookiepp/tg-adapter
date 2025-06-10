
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

class minmax:
	def __init__(self, values, indices):
		self._values = values
		self._indices = indices
	
	@property
	def values(self):
		return self._values
	
	@property
	def indices(self):
		return self._indices
		
	def __iter__(self):
		return iter( (self._values, self._indices) )

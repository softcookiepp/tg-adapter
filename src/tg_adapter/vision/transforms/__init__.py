from enum import Enum

class InterpolationMode(Enum):
	NEAREST = 1
	NEAREST_EXACT = 2
	BILINEAR = 3
	BICUBIC = 4
	BOX = 5
	HAMMING = 6
	LANCZOS = 7

from . import functional

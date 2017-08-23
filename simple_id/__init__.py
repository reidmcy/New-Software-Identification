try:
    import torch.cuda
except ImportError:
    raise ImportError("Pytorch not found. Go to http://pytorch.org for instructions on installing it.")

from .base import createClassifier, loadModel, analyseDF
from .neuralnet import BiRNN
from .utilities import compareRows, varsFromRow

__version__ = '0.0.1'

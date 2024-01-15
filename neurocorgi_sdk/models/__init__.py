# NeuroCorgi SDK, CeCILL-C license

# If n2d2 is installed in the user python environment
# then load n2d2 models
try:
    import n2d2
except:
    pass
else:
    from .n2d2 import *


# Torch models of NeuroCorgiNet
from .neurocorginet import *
from .neurocorginet_fq import *

# Import models for specific tasks
from .classification import *

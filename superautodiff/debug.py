import autodiff as sad
import numpy as np
import math

### Working script used solely to help debug certain functions and tests ###

x1 = sad.AutoDiff('x', 2*math.pi)
f = sad.cos(x1)

print(f)
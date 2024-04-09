import numpy as np
color = np.empty([480,640,3], dtype = np.uint8)
color[0,0,0] = 0
print(color.any())

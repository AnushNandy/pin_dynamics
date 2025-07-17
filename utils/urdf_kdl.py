import numpy as np
from scipy.spatial.transform import Rotation as R

xyz = np.array([
                0.375, 0, 0.11192, 
                0.178, 0, -0.037, 
                -0.045, 0, 0.097])

rpy = np.array([
                0,0,0,
                1.5708, 0.0, 1.5708,
                0, -1.5708, 0])

xyz = np.reshape(xyz, (3,3))
rpy = np.reshape(rpy, (3,3))

rpy_neocis = rpy.copy()
rpy_neocis[:, [2, 0]] = rpy[:, [0, 2]]

for i in range(len(xyz)):
    trans_i = xyz[i]
    rot_i = rpy[i]
    r = R.from_euler("xyz", rot_i, degrees=False)
    r = r.inv()
    
    trans_i_neocis = r.apply(trans_i)
    print(',\t'.join(map(str, trans_i_neocis)),",\t", ',\t'.join(map(str, rpy_neocis[i])), ",\t// link", i)
import numpy as np
def rrmse(img1, img2):
    return np.sqrt(np.sum((img1 - img2)**2)/np.sum(img1**2))

a = np.array([[2,3],[4,5]])
b = np.array([[1,2],[3,4]])

print(rrmse(a, b))

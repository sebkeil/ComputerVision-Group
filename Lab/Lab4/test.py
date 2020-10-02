
import numpy as np 

A = np.arange(1,37).reshape(6,6)
b = np.arange(1,7)
x = np.dot(np.linalg.inv(A),b)

T = np.arange(1,13).reshape(2,6) 

result = np.dot(T, x)
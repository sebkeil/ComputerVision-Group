
import numpy as np 

A = np.arange(1,37).reshape(6,6)
b = np.arange(1,7)
x = np.dot(np.linalg.inv(A),b)
xx = A * b

T = np.arange(1,13).reshape(2,6) 

result = np.dot(T, x)


shift = np.zeros((2,1))

import matplotlib.pyplot as plt 
import numpy as np 





f = open('check_lr.txt')

arr = []

for x in f:
	arr.append(float(x))


y = np.arange(len(arr))

plt.figure(figsize=(20,5))
plt.plot(y, arr, color='red')
plt.savefig('lr.png')
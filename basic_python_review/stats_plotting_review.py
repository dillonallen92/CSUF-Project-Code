import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

alpha = 0.5

x = np.linspace(0, 5, 1000)

y = x 

y_noise = x + alpha*np.random.randn(1000)

print(f"x size: {np.shape(x)}\n")
print(f"y size: {np.shape(y)}\n")
print(f"y_noise size: {np.shape(y_noise)}")

plt.figure()
plt.scatter(x, y_noise)
plt.plot(x,y, color="red")
plt.show()



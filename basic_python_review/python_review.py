# basic printing
print("Hello, World!")
name = "dillon"
print(f"Hello, {name}!")

# List comprehension
squares = [x**2 for x in range(1,11)]
print(squares)

# functions
def square_function(val):
  return val*val

print(f"The square of 4 is {square_function(4)}")

# Library specific code
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

sns.set_theme()
# x = np.linspace(-10, 10, 1000)
# y = np.sin(x)
# plt.plot(x,y)
# plt.show()

# seaborn tutorial
# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
plt.show()



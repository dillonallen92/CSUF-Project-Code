def my_func(items):
	items.append(4)

x = [1, 2, 3]
b = x.copy()

my_func(b)
print(x)

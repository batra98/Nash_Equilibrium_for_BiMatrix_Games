import numpy as np

rows = int(input())
cols = int(input())

random_matrix_array = np.random.randint(-100,100,size=(2*rows*cols))

print(rows)
print(cols)

for ele in random_matrix_array:
	print(ele,end=" ")
print()
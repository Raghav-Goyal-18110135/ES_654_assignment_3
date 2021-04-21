import numpy as np

arr = np.random.randint(1,10,(2,3))
print(arr)
for i in range(2):
    for j in range(3):
        print(arr[i][j])
    print()
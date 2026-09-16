import numpy as np

# # print(np.__version__)

# data = np.array([10,20,30])

# print(data)  #[10 20 30]
# print(data*2) #[20 40 60]
# print(type(data))  # <class 'numpy.ndarray'>
# print(data.shape)  #(3,) 3 elements in 1D array


# data2D = np.array([[1,2,3],[4,5,6]])

# print(data2D)  # [[1 2 3] [4 5 6]]
# print(data2D.shape)  #(2, 3) 2 rows and 3 columns
# print(data2D+2)  # [[3 4 5] [6 7 8]]
# print(np.max(data2D))  # 6
# print(np.min(data2D))  # 1
# print(np.mean(data2D))  # 3.5
# print(np.sum(data2D))  # 21
# print(np.std(data2D))  # 1.707825127659933

# print(data >=20)  # [False  True  True] boolean filtering
# print("\n", data2D >=2)  # [[False  True  True] [ True  True  True]] boolean filtering

# print(data2D[data2D >= 2])  # [2 3 4 5 6] filtering values greater than or equal to 2 boolean indexing

# print(np.sum(data[data >= 20]))  # 50 sum of values greater than or equal to 20

# print(np.sum(data>=20))  # 2 count of values greater than or equal to 20

# # Creating special Arrays

# print(np.zeros(5))  # [0. 0. 0. 0. 0.] 1D array of zeros
# print(np.zeros((2,3)))  # [[0. 0. 0.]   [0. 0. 0.]] 2D array of zeros
# print(np.ones(5))  # [1. 1. 1. 1. 1.] 1D array of ones
# print(np.ones((2,3)))  # [[1. 1. 1.]    [1. 1. 1.]] 2D array of ones
# print(np.arange(5))  # [0 1 2 3 4] 1D array of numbers from 0 to 4
# print(np.arange(1,5))  # [1 2 3 4] 1D array of numbers from 1 to 4
# print(np.arange(1,10,2))  # [1 3 5 7 9] 1D array of numbers from 1 to 9 with step size of 2

data = np.array([
    [10,20,30],
    [40,50,60]
])

print(np.mean(data, axis=0))  # [25. 35. 45.] mean of each column
print(np.mean(data, axis=1))  # [20. 50.] mean of each row

print(np.random.randint(0,101,5))  # random integer between 0 and 100, 5 numbers
print(np.random.randint(0,101,(2,3)))  # random integer between 0 and 100, 2 rows and 3 columns
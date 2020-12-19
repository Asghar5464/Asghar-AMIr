## Import the numpy package under the name np
import numpy as np
import random
## Create a null vector of size 10
x = np.zeros(10)
print (x)
## Create a vector with values ranging from 10 to 49
v = np.arange(10,49)
print(v)
## Find the shape of previous array in question 3
print(v.shape)
## Print the type of the previous array in question 3
print(v.dtype)
## Print the numpy version and the configuration
print(np.__version__)
print(np.show_config())
## Print the dimension of the array in question 3
print(v.ndim)
## Create a boolean array with all the True values
bool_arr = np.ones(10, dtype=bool)
print(bool_arr)
## Create a two dimensional array
array = np.arange(20).reshape(4,5)
print(array)
## Create a three dimensional array
array1 = np.arange(27).reshape(3,3,3)
print(array1)
## Reverse a vector (first element becomes last)
x = np.arange(10, 30)
print("Original array:")
print(x)
print("Reverse array:")
x = x[::-1]
print(x)
## Create a null vector of size 10 but the fifth value which is 1
null = np.zeros(10)
print(null)
print("Update sixth value to 11")
null[4] = 1
print(null)
## Create a 3x3 identity matrix
array_2D=np.identity(3)
print('3x3 matrix:')
print(array_2D)
## arr = np.array([1, 2, 3, 4, 5])
##Convert the data type of the given array from int to float
arr=np.array([1, 2, 3, 4, 5])
arr = arr.astype('float64') 
print(arr)
arr1 = np.array([[1., 2., 3.], [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
## Multiply arr1 with arr2
Multiply=arr1*arr2
print(Multiply)
##############################################
arr3 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr4 = np.array([[0., 4., 1.],[7., 2., 12.]])
## Make an array by comparing both the arrays provided above
print("Array arr3: ", arr3) 
print("Array arr4: ", arr4) 
  
print("arr3 > arr4") 
print(np.greater(arr3, arr4)) 
  
print("arr3 >= arr4") 
print(np.greater_equal(arr3,arr4)) 
  
print("arr3 < arr4") 
print(np.less(arr3, arr4)) 
  
print("arr3 <= arr4") 
print(np.less_equal(arr3, arr4)) 
############################################################
##Extract all odd numbers from arr with values(0-9)
a = np.arange(0,9)
arr5=a[a % 2 == 1]
print(arr5)
## Replace all odd numbers to -1 from previous array
arr5[0]=-1
arr5[1]=-1
arr5[3]=-1
arr5[2]=-1
print(arr5)
##############################################################
arr6 = np.arange(10)
##Replace the values of indexes 5,6,7 and 8 to 12
arr6[5]=12
arr6[6]=12
arr6[7]=12
arr6[8]=12
print(arr6)
## Create a 2d array with 1 on the border and 0 inside
x = np.ones((5,5))
print("Original array:")
print(x)
print("1 on the border and 0 inside in the array")
x[1:-1,1:-1] = 0
print(x)
##############################################################
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[1:2,1:2]=12
print(arr2d)
#################################################################
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
## Convert all the values of 1st array to 64
arr3d[0:1,0:]=64
print(arr3d)
## Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it
arr2d=np.arange(0,9).reshape(3,3)
print(arr2d[0:1,0:])
## Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it
print(arr2d[1:2,1:2])
## Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows
print(arr2d[0:2,2:])
## Create a 10x10 array with random values and find the minimum and maximum values
x = np.random.random((10,10))
print("Original Array:")
print(x) 
xmin, xmax = x.min(), x.max()
print("Minimum and Maximum Values:")
print(xmin, xmax)
#####################################################################################
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
## Find the common items between a and b
print(a[a==b])
######################################################################################
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
## Find the positions where elements of a and b match
v=np.searchsorted(a, np.intersect1d(a, b))
print(v)
############################################################################################
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
## Find all the values from array data where the values from array names are not equal to Will
print(data[names !='Will'])
## Find all the values from array data where the values from array names are not equal to Will and Joe
print(data[names == 'Bob'])
## Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.
arr = np.random.uniform(low=1.0, high=15.0, size=(5,3))
print(arr)
## Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.
a = np.random.uniform(low=1.0, high=16.0, size=(2,2,4))
## Swap axes of the array you created in Question 32
np.swapaxes(a, 0, 1)
print(a)
## Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0
import math
arr = np.arange(10)
print(arr)
new_arr = filter(lambda x: x if math.sqrt(x) > 0.5 else 0, arr)
np.asarray(list(new_arr))
## Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays
a = np.random.randint(12, size=random.randrange(0, 12))
b= np.random.randint(12, size=random.randrange(0, 12))
print(a)
print(b)
print(np.where(a == b))
#####################################################################################
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
## Find the unique names and sort them out!
print(np.unique(names))
#####################################################################################
a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
## From array a remove all items present in array b
val = np.intersect1d(a, b)

a = filter(lambda x: x if x not in val else None, a)
print(np.asarray(list(a)))
## Following is the input NumPy array delete column two and insert following new column in its place.
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 
sampleArray[:, 1] = 10
print(sampleArray)
#######################################################################################
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(np.dot(x, y))
## Generate a matrix of 20 random values and find its cumulative sum
matrix = np.random.randint(20, size=(random.randint(0, 5), random.randint(0, 5)))
print(np.cumsum(matrix))
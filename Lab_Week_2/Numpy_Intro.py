import numpy as np
import time

#Numpy routimes which allocates memory and fill array with value
a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros(4,) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(f"np.random.random_sample(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#NumPy routines which allocates memory and fill arrays with value but do not accept
#shape as input argument
a = np.arange(4.)
print(f"np.arange(4.) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a= np.random.rand(4)
print(f"np.random.rand(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.arange(10)
print(a)
print(a[2].shape, a[2])
print(a[-1])
try:
    c = a[10]
except Exception as e:
    print("The error is")
    print(e)

#Single vector operations
a = np.array([1,2,3,4])
print(a)
b= -a
print ("-a is", b)
b = np.sum(a)
print('sum of all elements in a is', b)
b = np.mean(a)
print('mean of all elements in a is',b)
b = a**2
print('double the elements in a is', b)

# Look for the some of the rest notes in notebook

a = np.array([1,2,3,4])
b = np.array([-1,4,3,2])
c = np.dot(a,b)
print(f"{c}, shape of c is {c.shape}" )

c = np.dot(b,a)
print(f"{c}, shape of c is {c.shape}" )

#Checking speed of numpy

def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

np.random.seed(1)
a = np.random.rand(10000000) #random arrays very large
b = np.random.rand(10000000)

tic = time.time() #start time
c = np.dot(a,b)
toc = time.time() #end time

print(f"np.dot(a,b) = {c:.4f}")
print(f"time taken: {1000*(toc-tic):.4f} ms")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b) #remove these arrays from memory

#Matrices

a = np.zeros((2,5)) #(m,n) where m rows and n columns
print(f"a shape = {a.shape}, a = {a}")  

a = np.random.random_sample((1,1))
print(f"a shape = {a.shape}, a = {a}")   

#reshape is a convenient way to create matrices
a = np.arange(6).reshape(-1,2)
print(f"a shape = {a.shape}, a = {a}")   
print(a[2,0]) #an element
print(a[2]) #a row

#slicing
#[row , column] in the format of (start:stop:step)
a = np.arange(20).reshape(-1,10)
print(f"a shape = {a.shape}, a = {a}")  
#access 5 consecutive elements 
#
print(a[0, 2:7:10])

#access 5 consecutive elements (start:stop:step) in two rows
print(a[:, 2:7:10])

#access all elements
print(a[:,:])

#access all elements in one row
print(a[1,:])
#or
print(a[1])
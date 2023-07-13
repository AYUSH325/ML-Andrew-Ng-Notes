import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)

# Problem Statement
# You will use the motivating example of housing price prediction. 
# The training dataset contains three examples with four features 
# (size, bedrooms, floors and, age) 

# Size (sqft)	Number of Bedrooms	Number of floors	Age of Home	Price (1000s dollars)
# 2104	                5	               1	            45	                460
# 1416	                3	               2	            40	                232
# 852	                2	               1	            35	                178
# You will build a linear regression model using these values so you can then predict the price for other houses. 
# For example, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.

# def predict_single_loop(x, w, b):
#     """
#     single predict using linear regression
    
#     Args:
#       x (ndarray): Shape (n,) example with multiple features
#       w (ndarray): Shape (n,) model parameters    
#       b (scalar):  model parameter     
      
#     Returns:
#       p (scalar):  prediction
#     """
#     n = x.shape[0]
#     p = 0
#     for i in range(n):
#         p_i = x[i] * w[i]
#         p = p + p_i
#     p = p + b
#     return p

def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x,w) + b
    return p

# Compute Cost With Multiple Variables
def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(w, X[i]) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

#Compute Gradient with Multiple Variables



X_train = np.array([[2104, 5, 1, 45], [1416,3,2,40], [852,2,1,35]])
y_train = np.array([460, 232, 178])

print(f"X Shape:{X_train.shape}, X Type:{type(X_train)}")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

#get a row from training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

#make a prediction
f_wb = predict(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#Compute and display cost with pre-chosen optimal parameters
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')






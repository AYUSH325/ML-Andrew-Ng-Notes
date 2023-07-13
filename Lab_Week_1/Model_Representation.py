import matplotlib.pyplot as plt
import numpy as np

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
        x (ndarray (m,)): Data, m examples
        w,b (scalar)    : model parameters
    """
    m = x.shape[0]
    #np.zero(n) will return a one-dimensional numpy array with  𝑛  entries
    f_wb = np.zeros(m) 
    for i in range(m):
        f_wb[i] = w * x[i] + b
    
    return f_wb

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")          

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

#Training example x_i, y_i
i = 0  # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i}) = ({x_i}, {y_i}))")

#Plotting the data

# Plot the data points
#The scatter() function arguments marker and c show the points as red crosses 
# (the default is blue dots).
plt.scatter(x_train, y_train, marker='x', c='r')
#Set the title
plt.title("Housing Prices")
#Set the y-axis label
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel('Size (1000 sqft)')
plt.show()

#Model Function
w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

tmp_f_wb = compute_model_output(x_train, w, b)

#Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

#Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

#Set the title
plt.title("Housing Prices")
#Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
#Set the x-axis label
plt.xlabel("Size (1000 sqft)")
plt.legend()
plt.show()

x_i = 1.2
cost_1200sqft = w * x_i +b

print(f"${cost_1200sqft:.0f} thousand dollars")
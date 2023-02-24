import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#-----------Data--------------

# Original Data
X = np.array([[1],[3],[5],[7]]) #features (inputs)
y = np.array([2,4,7,8]) #labels (outputs)

# Spliting Data
X_train, X_test, y_train, y_test = train_test_split(X,y)

#--------Algorithm-----------
model = LinearRegression() #Finds the straight line through the data

#----------Learning/Training----------
model.fit(X_train, y_train) #Learns slope (m) and the y-itercept (b). Scikit-learn calls these coef_ and intercept_
m = model.coef_
b = model.intercept_ 

#------------Predicting/Testing---------------
y_test_hat = model.predict(X_test)
print(y_test_hat)

#--------------Plotting-------------
# Plotting training and testing data
plt.scatter(X_train,y_train, c = 'b', label = 'Training Data') 
plt.scatter(X_test, y_test, c = 'r', label = 'Test Data')
plt.legend()
plt.grid()
plt.axis('Equal')
plt.ylabel('Height (Output)')
plt.xlabel('Weight (Input)')

# Predicted Data
plt.scatter(X_test,y_test_hat, c = 'orange', label = 'Prediction')

# Plotting the predicted line that has been learned by the Ml algorithm.
x_axis = np.array([1,7])
y_axis = m * x_axis + b 
plt.plot(x_axis,y_axis) 

# Plot the error
plt.text(X_test, (y_test + y_test_hat)/2, s = 'Error')
error = y_test - y_test_hat
plt.arrow(X_test[0,0],y_test_hat[0], 0, error[0])




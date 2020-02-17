from random import uniform
import numpy as np
import matplotlib.pyplot as plt

n=100
#Define arrays used in algorithm#
x_train=np.zeros(shape=(n,2))
a=np.zeros(shape=(n,1))
b=np.zeros(shape=(n,1))
for row in range(0,n):
    x_train[row]=(uniform(-1,1),uniform(-1,1))

#Define target function#
def target(x):
    return np.square(x)

#Calculate the line going through the two entries on each row of x#
for slope in range(0,n):
    a[slope]=(target(x_train[slope][0])-target(x_train[slope][1]))/(x_train[slope][0]-x_train[slope][1])

for intercept in range(0,n):
    b[intercept]=target(x_train[intercept][0])-(a[intercept]*x_train[intercept][0])    


del(slope,intercept,row)

#Get the average hypothesis function#
slope=np.mean(a)
y_intercept=np.mean(b)

def g(a,x,b):
    return(a*x+b)

def g_bar(x):
    return (slope*x+y_intercept)

#Get bias and variance of our model#
bias=0
variance=0

x_test=np.zeros(shape=(n,2))
for row in range(0,len(x_test)):
    x_test[row]=(uniform(-1,1),uniform(-1,1))

for row in range (0,len(x_test)):
    for col in range(0,1):
        bias += np.square(g_bar(x_test[row][col])-target(x_test[row][col]))/n
        variance += np.square(g(a[row],x_test[row][col],b[row])-g_bar(x_test[row][col]))/n

del row

#Plot g_bar and the target function#
xvals = np.arange(-1,1,.01)
yvals_quadratic=target(xvals)
plt.plot(xvals,yvals_quadratic,'k-',linewidth=2)
yvals_linear = g_bar(xvals)
plt.plot(xvals,yvals_linear,'k-',linewidth=2)
plt.grid()
plt.show()



















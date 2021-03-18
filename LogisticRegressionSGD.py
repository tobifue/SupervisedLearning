##FDA lab assignment. Tobias Fuehles, 11936325

##import the relevant libraries/packages for the assignment.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def conv(s):
    """Converter function to map string value "Iris-setosa" with int 1 
       and remaining values with 0.
    """
    if s == "Iris-setosa":
            s = 1
    else:
            s = 0
    return s

#Reading in the .csv file by using the converter function 
#and to prevent loading in as bytes format encoding is specifieed as utf8.
data = np.loadtxt("lab_iris_data.csv", delimiter = ",", usecols = (0,1,2,3), 
                  converters = {3: conv},encoding="utf8")


##Visualize the data.
x = data[:,3]
col = np.where(x<1,"r","g")
#Plot the 3D figure by defining the X1, X2 and X3 coordinates 
#as given from the .csv file.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
color = ["red","green"]
#Plot first 50 entries green and last 50 entries red.
ax.scatter(data[:50,0],data[:50,1],data[:50,2], c="g", 
           label = "class label 1")
ax.scatter(data[50:100,0],data[50:100,1],data[50:100,2], c="r", 
           label = "class label 0")
#Set limit of the axes.
ax.set_xlim(0,7)
ax.set_ylim(0,5)
ax.set_zlim(0,5)
#Set the axes labels.
ax.set_xlabel("$X1$")
ax.set_ylabel("$X2$")
ax.set_zlabel("$X3$")
#Plot legend.
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
#Save figure for further use.
plt.savefig("FDATaskOne.pdf")


##Extract and stack sample values and add bias as X4.
X1 = [item[0] for item in data]
X1 = [item[0] for item in data]
X2 = [item[1] for item in data]
X3 = [item[2] for item in data]
X4 = np.ones((1,100))
X = np.vstack((X1,X2,X3,X4))

##Extract label values.
Y = np.asarray(([item[3] for item in data]))


##Create an index to be able to select random samples and labels. 
#Could be X or Y as input parameter.
def create_ind(Y):
    index = []
    for i in range(len(Y)):
        index.append(i)
    return index


def logistic(theta, samples):
    """Calculate hypothesis of logistic regression, 
    determining probability of a sample x(i) being in class 0 or 1.
    """
    probability = 1 / (1 + np.exp(-samples.T*theta))
    return probability


def grad(theta, samples, labels, reg_factor):
    """Calculate the gradient of regularized loss function.
    """
    update_ = ((logistic(theta, samples) - labels.T).T*samples.T).T
    update = update_+(reg_factor*theta)
    return update

    
def train_classifier(num_iter, learn_rate, batch_size):
    """Calculate an optimized classifier iterated over number of iterations. 
       The calculation also depends on learning rate and batch size.
    """
    theta = np.zeros((4,1))
    #reg_factor could also be an input of train_classifier(...,reg_factor)
    reg_factor = 1
    theta_lst = []
    learn_rate = learn_rate
    index = create_ind(Y)
    for t in range(num_iter):
        t += 1
        learn_rate_upd = learn_rate/np.sqrt(t)
        ind_ = np.random.choice(index,batch_size)
        samples = np.asmatrix(np.array([X[0][ind_],X[1][ind_],
                                        X[2][ind_],X[3][ind_]]))
        labels = np.asmatrix(Y[ind_])
        theta_upd = theta - (learn_rate_upd * (1/batch_size)
                          * grad(theta, samples, labels, reg_factor))
        theta = theta_upd
        theta_lst.append(theta)
    return theta_lst
      
      
##Set theta to calculate loss_fct. Could also be set in the loss_fct function
##but makes it easier to experiment with different or specific parameters.
theta = train_classifier(100,0.1,20)


def loss_fct(samples, labels, reg_factor):
    """Determine the loss according to the determined most updated theta value.
    sum_1 consists of the loss over all sample and label values and sum_2 of
    the regularization term.
    """
    loss = []
    loss_sum = []
    for t in range(len(theta)):
      for i in range(len(Y)):
        #Calculate the loss over samples and labels.
        sum_1 = ((-labels[i]*np.log(logistic(theta[t], 
                 np.asmatrix(np.array([X[0][i],X[1][i],X[2][i],X[3][i]])).T))
               - (1-labels[i])*np.log(1-logistic(theta[t], 
                 np.asmatrix(np.array([X[0][i],X[1][i],X[2][i],X[3][i]])).T)))) 
        loss.append(sum_1)
        #Add the sum of the regularization term.
        sum_2 = reg_factor/2*(np.power(theta[t][0],2) 
               + np.power(theta[t][1],2) 
               + np.power(theta[t][2],2) + np.power(theta[t][3],2))
      loss_sum.append(sum(loss)+sum_2)
      loss = []
    return loss_sum


##Plot the loss over the number of iteratinos.
#Set x axis as number of iterations.
y_axis = np.asarray(range(len(theta)))
y_axis = y_axis +1
#Set y axis as amount of loss.
loss = loss_fct(X,Y,1)
loss = list(np.ravel(loss))
fig, ax = plt.subplots()
ax.plot(y_axis, loss, label='$loss$ $over$ $theta_t$')
legend = ax.legend(loc='upper center', shadow=True)
plt.ylabel("$loss$")
plt.xlabel("$t$")
plt.show
plt.savefig("FDATaskTwo.pdf")


##Separating hyperplane
#Define evenly spaced X1 and X2 coordinates (by default of np.linspace n = 50)
x = []
x = np.linspace(0,7)
y = []
y = np.linspace(0,5)

##Create an array of all possible combination points of X1 and X2
X1,X2 = np.meshgrid(x,y)


def calc_X3(xxx,yyy):
    """Calculate the X3 values with X1 and X2 to determine the hyperplane. 
       Formula results by setting logistic function equal with 0.5 
       and resolving for X3.
    """
    X3_lst = []
    for i in range(len(np.ravel(X1))):
            z = -((theta_upd[3]+xxx[i]*theta_upd[0]+yyy[i]*theta_upd[1])
                  / theta_upd[2])
            X3_lst.append(z)
    return X3_lst


##Select most iterated theta value to determine X3
theta_upd = theta[len(theta)-1]
##Store results from calc_X3 for all possible X1/X2 combinations as array.
X3_ = np.array(calc_X3(np.ravel(X1),np.ravel(X2)))
##Store X3 coordinates in same format as X1 and X2.
X3 = X3_.reshape(X1.shape)


##Visualize the date including the separating hyperplane.
#For loop to receive figures from different angles.
for angle in range(0,360):
    x = data[:,3]
    col = np.where(x<1,"r","g")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:50,0],data[:50,1],data[:50,2], c="g", #
               label = "class label 1")
    ax.scatter(data[50:100,0],data[50:100,1],data[50:100,2], c="r", 
               label = "class label 0")
    ax.set_xlabel("$X1$")
    ax.set_ylabel("$X2$")
    ax.set_zlabel("$X3$")
    #Plot the surface of the hyperplane with X1, X2 and calculated X3.
    ax.plot_surface(X1, X2, X3, alpha = 1)
    legend = ax.legend(loc='upper right', shadow=True, fontsize='x-small')
    #Store 360 .pdf files with images from all angles.
    plt.savefig("FDA" + str(angle) +".pdf")
    #Set initial angle and change angle per iteration by angle.
    ax.view_init(30,angle)
    plt.draw()
    plt.pause(0.001)
    

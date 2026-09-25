<H3>NAME: Subhash V</H3>
<H3>REG. NO.: 212224240163</H3>
<H3>EX. NO.3</H3>
<H3>DATE: 18.08.2026</H3>
<H2 aligh = center> Implementation of MLP for a non-linearly separable data</H2>
<h3>Aim:</h3>
To implement a perceptron for classification using Python
<H3>Theory:</H3>
Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows:

XOR truth table
![Img1](https://user-images.githubusercontent.com/112920679/195774720-35c2ed9d-d484-4485-b608-d809931a28f5.gif)

XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below

![Img2](https://user-images.githubusercontent.com/112920679/195774898-b0c5886b-3d58-4377-b52f-73148a3fe54d.gif)

The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.To separate the two outputs using linear equation(s), it is required to draw two separate lines as shown in figure below:
![Img 3](https://user-images.githubusercontent.com/112920679/195775012-74683270-561b-4a3a-ac62-cf5ddfcf49ca.gif)
For a problem resembling the outputs of XOR, it was impossible for the machine to set up an equation for good outputs. This is what led to the birth of the concept of hidden layers which are extensively used in Artificial Neural Networks. The solution to the XOR problem lies in multidimensional analysis. We plug in numerous inputs in various layers of interpretation and processing, to generate the optimum outputs.
The inner layers for deeper processing of the inputs are known as hidden layers. The hidden layers are not dependent on any other layers. This architecture is known as Multilayer Perceptron (MLP).
![Img 4](https://user-images.githubusercontent.com/112920679/195775183-1f64fe3d-a60e-4998-b4f5-abce9534689d.gif)
The number of layers in MLP is not fixed and thus can have any number of hidden layers for processing. In the case of MLP, the weights are defined for each hidden layer, which transfers the signal to the next proceeding layer.Using the MLP approach lets us dive into more than two dimensions, which in turn lets us separate the outputs of XOR using multidimensional equations.Each hidden unit invokes an activation function, to range down their output values to 0 or The MLP approach also lies in the class of feed-forward Artificial Neural Network, and thus can only communicate in one direction. MLP solves the XOR problem efficiently by visualizing the data points in multi-dimensions and thus constructing an n-variable equation to fit in the output values using back propagation algorithm

<h3>Algorithm :</H3>

Step 1 : Initialize the input patterns for XOR Gate<BR>
Step 2: Initialize the desired output of the XOR Gate<BR>
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron  and 1 output neuron<BR>
Step 3: Repeat the  iteration  until the losses become constant and  minimum<BR>
    (i)  Compute the output using forward pass output<BR>
    (ii) Compute the error<BR>
	(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm. <BR>
    (iv) Modify the weight as per delta rule.<BR>
    (v)  Append the losses in a list <BR>
Step 4 : Test for the XOR patterns.

<H3>Program:</H3>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
  def __init__(self,learning_rate=0.1):
    self.learning_rate = learning_rate
    self.b=0.0
    self.w=None
    self.misclassified_samples=[]

  def fit(self,x:np.array,y:np.array,n_iter=10):
    self.b=0.0;
    self.w=np.zeros(x.shape[1])
    self.misclassified_samples=[]

    for _ in range(n_iter):
      errors=0
      for xi,yi in zip(x,y):
        update = self.learning_rate*(yi-self.predict(xi))
        self.b+=update
        self.w+=update*xi
        errors+=int(update!=0.0)
      self.misclassified_samples.append(errors)

  def f(self,x:np.array)->float:
    return np.dot(x,self.w)+self.b

  def predict(self,x:np.array):
    return np.where(self.f(x)>= 0,1,-1)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df=pd.read_csv(url,header=None)
print(df.head())

df.describe()

y=df.iloc[:,4].values
x=df.iloc[:,0:3].values

fig=plt.figure()
ax=plt.axes(projection='3d')
ax.set_title("Iris dataset")
ax.set_xlabel("sepal_length (cm)")
ax.set_ylabel("sepal_width (cm)")
ax.set_zlabel("petal_length (cm)")
ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red', marker='o', s=4, label="Iris Setosa")
ax.scatter(x[50:100,0],x[50:100,1],x[50:100,2],color='blue',marker='^',s=4,label="Iris Versicolor")
ax.scatter(x[100:150,0],x[100:150,1],x[100:150,2],color='green',marker='x',s=4,label="Iris Virginica")
plt.legend(loc='upper left')
plt.show()

x=x[0:100,0:2]
y=y[0:100]

plt.figure(figsize=(10,6))
plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()

y=np.where(y=='Iris-setosa',1,-1)
x[:,0] = (x[:,0]-x[:,0].mean())/x[:,0].std()
x[:,1] = (x[:,1]-x[:,1].mean())/x[:,1].std()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train,y_train)

print("Accuracy:", accuracy_score(classifier.predict(x_test), y_test) * 100)

plt.figure(figsize=(4, 4))
plt.plot(range(1, len(classifier.misclassified_samples) + 1), classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()
```

<H3>Output:</H3>

<img width="492" height="440" alt="image" src="https://github.com/user-attachments/assets/23e2e894-2ad3-4a36-b4ea-b6c0746cc16e" />


<H3> Result:</H3>
Thus, XOR classification problem can be solved using MLP in Python 

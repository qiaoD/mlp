'''
@ author : QD
@ time   : 2018/4/23

# a nerual network : [2, 3, 2]

## tips:

1. I will store the matrixs using FP in every step of computer graph and
use them in BP to value diff.
2. The training alg is `BP`.
3. activation function : `tanh`.
4. Output function : `softmax`.
5. Loss function : `Cross-Entropy : L = sum of  y*lna`
6. training process : `BGD`

[array([[ 0.92816437,  1.84122705,  0.12902971],
       [-2.17619744, -0.15147832, -0.21606233]]),
       array([[-1.23350574,  0.50555409,  1.36854024,  1.51863461],
       [-1.08543991, -1.61453349,  1.6414295 ,  1.58735763],
       [ 0.81262234,  2.31931108, -0.94774043,  0.38069249]]),
       array([[-0.06373717,  0.93321665,  1.37087378, -0.44450204,  0.56296937],
       [-0.57696044, -0.33489326,  0.04421253, -0.57622581,  0.63734225],
       [ 0.76863782, -0.54040799, -0.23114118,  1.86318457,  0.56687331],
       [ 1.17174262,  1.39559318,  1.44479624, -0.01291493, -1.88035528]]),
       array([[ 0.59947793,  0.52710486],
       [-2.0119332 ,  1.47074829],
       [-2.00716691,  1.93870404],
       [-0.28935927, -1.33817128],
       [-0.02643005, -1.57186956]])]


'''



import numpy as np
import matplotlib.pyplot as plt

import sklearn
import sklearn.datasets
import sklearn.linear_model

# activation function

class Tanh:
    def forward(self, x):
        return np.tanh(x)

    def backward(self, wxb, dz):
        return (1 - self.forward(wxb) ** 2) * dz

class Sigmoid:
    def forward(self, ):
        pass

    def backward(self, ):
        pass


# Output layer

class Softmax:
    def predict(self, x):
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis = 1, keepdims = True) # keep the matrixs Dim

    def loss(self, inp, y):
        num = inp.shape[0]
        probs = self.predict(inp)
        logprobs = -np.log(probs[range(num), y])
        data_loss = np.sum(logprobs)
        return 1./num * data_loss

    def diff(self, x, y):
        num = x.shape[0]
        probs = self.predict(x)
        probs[range(num), y] -= 1
        return probs


# Gate in computer graph

class MultiplyGate:
    def forward(self, W, x):
        return np.dot(x, W)

    def backward(self, x, w, dz):
        dw = np.dot(np.transpose(x), dz)
        dx = np.dot(dz, np.transpose(w))
        return dw, dx

class AddGate:
    def forward(self, wx, b):
        return wx + b

    def backward(self, wx, b, dz):
        # BGD so db的变化量为所有训练集样本的变化量之和
        db   = np.dot(np.ones((1,dz.shape[0])), dz)
        dmul = np.ones_like(wx) * dz    # all 1
        return db, dmul


# get from github
def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)




class NN:
    def __init__(self, dim):
        self.layers_dim = len(dim)
        self.W = []  # Weight matrixs
        self.b = []  # bais matrixs

        # radom init w and b
        for i in range(self.layers_dim - 1):
            self.W.append(np.random.randn(dim[i],dim[i+1]))
            self.b.append(np.random.randn(dim[i+1]).reshape(1, dim[i+1]))

        # test
        self.w = [np.array([[ 0.92816437,  1.84122705,  0.12902971],
               [-2.17619744, -0.15147832, -0.21606233]]),
               np.array([[-1.23350574,  0.50555409,  1.36854024,  1.51863461],
               [-1.08543991, -1.61453349,  1.6414295 ,  1.58735763],
               [ 0.81262234,  2.31931108, -0.94774043,  0.38069249]]),
               np.array([[-0.06373717,  0.93321665,  1.37087378, -0.44450204,  0.56296937],
               [-0.57696044, -0.33489326,  0.04421253, -0.57622581,  0.63734225],
               [ 0.76863782, -0.54040799, -0.23114118,  1.86318457,  0.56687331],
               [ 1.17174262,  1.39559318,  1.44479624, -0.01291493, -1.88035528]]),
               np.array([[ 0.59947793,  0.52710486],
               [-2.0119332 ,  1.47074829],
               [-2.00716691,  1.93870404],
               [-0.28935927, -1.33817128],
               [-0.02643005, -1.57186956]])]



    def predict(self, x):
        Mulg = MultiplyGate()
        Addg = AddGate()
        actFunction = Tanh()
        Output = Softmax()
        inp = x
        for i in range(self.layers_dim - 1):
            mul = Mulg.forward(self.W[i], inp)
            add = Addg.forward(mul, self.b[i])
            inp = actFunction.forward(add)
        # need modify,output just a value<1
        probs = Output.predict(inp)
        return np.argmax(probs, axis=1)

    def loss(self, x, y):
        Mulg = MultiplyGate()
        Addg = AddGate()
        actFunction = Tanh()
        Output = Softmax()

        inp = x
        for i in range(self.layers_dim - 1):
            mul = Mulg.forward(self.W[i], inp)
            add = Addg.forward(mul, self.b[i])
            inp = actFunction.forward(add)

        return Output.loss(inp, y)


    def train(self, x, y, num_passes, epsilon, print_loss = False):

        for epoch in range(num_passes):
            # FP
            Mulg = MultiplyGate()
            Addg = AddGate()
            actFunction = Tanh()
            Output = Softmax()

            inp = x
            forwardResult = [(None, None, inp)]
            for i in range(self.layers_dim - 1):
                mul = Mulg.forward(self.W[i], inp)
                add = Addg.forward(mul, self.b[i])
                inp   = actFunction.forward(add)
                forwardResult.append((mul, add, inp))

            # BP
            dtanh = Output.diff(forwardResult[self.layers_dim-1][2], y)
            for i in range(len(forwardResult)-1, 0, -1):

                dadd = actFunction.backward(forwardResult[i][1], dtanh)
                # get the db
                db, dmul = Addg.backward(forwardResult[i][0], self.b[i-1], dadd)
                # get the dw
                dw, dtanh = Mulg.backward(forwardResult[i-1][2], self.W[i-1], dmul)

                self.W[i-1] += -epsilon * dw
                self.b[i-1] += -epsilon * db

            if (print_loss and epoch%100 == 0):
                print("Loss after iteration %i: %f" %(epoch, self.loss(x, y)))

        print(self.W)
        print(self.b)

if __name__ == '__main__':
    x = np.array([[1, 1],[-1, 1],[-1,-1],[1,-1]])
    y = np.array([0,1,0,1])
    #x,y = sklearn.datasets.make_multilabel_classification(n_classes=2, n_labels=1,allow_unlabeled=True,random_state=1)
    dim = [2, 3,4,5, 2] # 2 input 3 hidden 2 output
    dnn = NN(dim)
    # dnn.predict(x)
    dnn.train(x, y,num_passes=100000, epsilon=.01, print_loss=True)
    print(dnn.predict(x = np.array([0.01, 0.01])))

    # Plot the decision boundary
    plot_decision_boundary(lambda x: dnn.predict(x), x, y)
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def z(x):
    """Evaluate the sigmoid function at x."""
    return 1.0/(1.0 + np.exp(-x))


def h(Theta, X):
    """Evaluate the sigmoid function at each element of <Theta,X>."""
    return np.array([z(np.dot(Theta, x)) for x in X])


def gradient(Theta, X, Y):
    """Compute the gradient of the log-likelihood of the sigmoid."""
    pX = h(Theta, X) # i.e. [h(x) for each row x of X]
    return np.dot((Y - pX), X)


def logfit(X, Y, alpha=1, itr=10):
    """Perform a logistic regression via gradient ascent."""
    Theta = np.zeros(X.shape[1])
    for i in range(itr):
        Theta += alpha * gradient(Theta, X, Y)
    return Theta


def normalize(X):
    """Normalize an array, or a dataframe, to have mean 0 and stddev 1."""
    return (X - np.mean(X, axis=0))/(np.std(X, axis=0))


def tprfpr(P, Y):
    """Return the False Positive Rate and True Positive Rate vectors of the given classifier."""
    Ysort = Y[np.argsort(P)[::-1]]
    ys = np.sum(Y)
    tpr = np.cumsum(Ysort)/ys # [0, 0, 1, 2, 2, 3,..]/18
    fpr = np.cumsum(1-Ysort)/(len(Y)-ys)
    return (tpr, fpr)


def auc(fpr, tpr):
    """Compute the Area Under the Curve (AUC) given vectors of
    false positive rate and true positive rate"""
    return(np.diff(tpr) * (1 - fpr[:-1])).sum()

def run(array):
    train_file=array[0]
    
    ### Training #######
   
    train_data=pd.read_csv(train_file)
    train_data.dropna(inplace=True)
    X_train=np.ones((len(train_data),len(train_data.columns)))
    X_train[:,1:]=normalize(train_data.iloc[:,0:-1].values)
    Y_train = train_data.iloc[:,-1].values
    th=logfit(X_train, Y_train, alpha=0.01,itr=10)
    print (th)
    P=h(th,X_train)
    pd.crosstab(P > 0.2, Y_train)
    tpr, fpr = tprfpr(P, Y_train)
    plt.plot(fpr, tpr) # ROC curve
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('1772576-roc.png')
    print("AUC = ", auc(fpr, tpr)) # Area Under the Curve

    ######### Test #######

    test_file=array[1]
    test_data=pd.read_csv(test_file)
    test_data.dropna(inplace=True)
    X_test=np.ones((len(test_data),len(test_data.columns)+1))
    X_test[:,1:]=normalize(test_data.values)
    print ([format(i,'.4f') for i in h(th,X_test)])

arguments=sys.argv[1:]
run(arguments)

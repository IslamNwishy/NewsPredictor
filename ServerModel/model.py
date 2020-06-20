#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import numpy as np
from numpy import genfromtxt
import re






#class
class MultiClassLogisticRegression:

    def test(self):
        print("test here yay")
    
    #constructor
    def __init__(self,y, n_iter = 10000, weights_file="ServerModel/weights.csv", thres=1e-5):
        #set number of iterations
        self.n_iter = n_iter
        #set completion threshold of a batch (used to speed up the training process)
        self.thres = thres
        #set weights from a given file
        self.weights = np.genfromtxt(weights_file ,delimiter=",",dtype=float)
        self.classes = np.unique(y)
        self.class_labels = {c:i for i,c in enumerate(self.classes)}

    #fit the data to the model
    def fit(self, X, y, batch_size=10, lr=0.003, weights_file="weights.csv", rand_seed=42,verbose=False): 
        np.random.seed(rand_seed)   #set the same random values everytime
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.fit_data(X, y, batch_size, lr, verbose)
        np.savetxt(weights_file, self.weights, delimiter=",") #save values back to the weights file
        return self
 
    #fit a batch of size batch_size
    def fit_data(self, X, y, batch_size, lr, verbose):
        i = 0
        while (not self.n_iter or i < self.n_iter):
            #calculate loss and error
            self.loss.append(self.cross_entropy(y, self.__predict(X))) 
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.__predict(X_batch)
            #update weights
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            
            if np.abs(update).max() < self.thres: break  #if less than threshold then it is good enough
            
            #print progress
            if i % 1000 == 0 and verbose: 
                print(' Training Accuray at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i +=1
    
    #returns the weights given to each class for the input X
    def predict(self, X):
        return self.__predict(self.add_bias(X))
    
    #private utility for predict
    def __predict(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)
    
    #softmax function
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    #returns the predicted class given an input X
    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
  
    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)
    
    #encodes the label using one hot encoding
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.__predict(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))
    
    #reset weights to 0
    def reset_weights(self,x,y):
        self.weights=np.zeros(shape=(np.unique(y),x.shape[1]))

    def predict_cat(self, title, tfidf):
        cat_names = {0 : 'Business and Politics', 3 : 'Science and Technology', 1 : 'Entertainment', 2 : 'Health'}
        title = self._normalize_text(title)
        title= tfidf.transform([title]).toarray()
        cod = self.predict_classes(title)
        prop = self.predict(title)

        return cat_names[cod[0]], prop[0][cod[0]]
    
    def _normalize_text(self,s):
        #lower case everything
        s = s.lower()

        #remove punctuation, exclude word related ones
        s = re.sub('\s\W',' ',s)
        s = re.sub('\W\s',' ',s)

        #remove double spaces
        s = re.sub('\s+',' ',s)

        return s



# In[10]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





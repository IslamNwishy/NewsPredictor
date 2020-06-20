from ServerModel.model import MultiClassLogisticRegression
import numpy as np
import pandas as pd
import re
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder


class Server_hidden:


    # evaluate the data and update the weights if it gives better metrics
    def test(self,tfidf,encoder,log_reg,log_reg1):
        data = pd.read_csv("test_data.csv")
        data['TEXT'] = [_normalize_text(s) for s in data['TITLE']]

        test_data_x= tfidf.transform(data['TEXT'])
        test_data_x=test_data_x.toarray()

        encoder = LabelEncoder()
        y = encoder.transform(data['CATEGORY'])


        rr=log_reg.predict_classes(test_data_x)
        rr1=log_reg1.predict_classes(test_data_x)
        print("Accuracy score: ", accuracy_score(y, rr))
        print("Recall score: ", recall_score(y, rr, average = 'weighted'))
        print("Precision score: ", precision_score(y, rr, average = 'weighted'))
        print("F1 score: ", f1_score(y, rr, average = 'weighted'))

        old=0
        new=0

        if(accuracy_score(y, rr)>accuracy_score(y, rr1)):
            new+=1
        else:
            old+=1

        if(recall_score(y, rr, average = 'weighted')>recall_score(y, rr1, average = 'weighted')):
            new+=1
        else:
            old+=1   
        
        if(precision_score(y, rr, average = 'weighted')>precision_score(y, rr1, average = 'weighted')):
            new+=1
        else:
            old+=1
        
        if(f1_score(y, rr, average = 'weighted')>f1_score(y, rr1, average = 'weighted')):
            new+=1
        else:
            old+=1

        if(new>old):
            new_weights = np.genfromtxt(new_weights_file ,delimiter=",",dtype=float)
            np.savetxt("ServerModel/weights.csv", new_weights, delimiter=",") 
            print("Weights Updated!")
        else:
            print("Weights Did not Change")
        

        return


    def train(self, N):

        print('Begin Training ', N, ' New Items ...')
        new_data_file= "new_data.csv"
        old_data_file= "ServerModel/uci-news-aggregator.csv"
        new_weights_file= "new_weights.csv"
        old_weights_file= "weights.csv"

        

        def _normalize_text(s):
            #lower case everything
            s = s.lower()
            
            #remove punctuation, exclude word related ones
            s = re.sub('\s\W',' ',s)
            s = re.sub('\W\s',' ',s)
            
            #remove double spaces
            s = re.sub('\s+',' ',s)
            
            return s

        #Optain new Training data
        data = pd.read_csv(new_data_file)
        data['TEXT'] = [_normalize_text(s) for s in data['TITLE']]

        news= pd.read_csv(old_data_file)
        news['TEXT'] = [_normalize_text(s) for s in news['TITLE']]

        full_data=pd.concat([data, news])
        tfidf1 = TfidfVectorizer()

        # recaclulate tfidf
        x = tfidf1.fit_transform(full_data['TEXT'])
        
        tfidf = TfidfVectorizer(vocabulary=tfidf1.vocabulary_)

        #fit the new data
        x= tfidf.fit_transform(data['TEXT'])

        encoder = LabelEncoder()
        y = encoder.fit_transform(full_data['CATEGORY'])


        #update weights
        
        weights = np.genfromtxt( old_data_file ,delimiter=",",dtype=float)
        Added_y= x.shape[1]-weights.shape[1]

        added_weights= np.zeros((4,Added_y+1))


        new_weights = np.append(weights,added_weights,1)

        np.savetxt(new_weights_file, new_weights, delimiter=",") 


        log_reg= MultiClassLogisticRegression(y, weights_file= new_weights_file)
        log_reg1= MultiClassLogisticRegression(y, weights_file= old_weights_file)

        y_new = encoder.transform(data['CATEGORY'])


        #Train 10 Patches of size N/10
        l1=0
        inc = int (N/10)
        l2 = inc
        for i in range(10):
            print(l1,":",l2)
            ss=x[l1:l2]
            ss=ss.toarray()
            log_reg.fit(ss,y[l1:l2],lr=0.003, verbose=True,weights_file=new_weights_file)
            l1+=inc
            l2+=inc
        
        reset=np.array([["TITLE","CATEGORY"]])
        np.savetxt(new_data_file,rest, delimiter=",") #remove trained data
        print ("Training Completed!!")

        self.test(tfidf,encoder,log_reg,log_reg1)

        return


   
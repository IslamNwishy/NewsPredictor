
from ServerModel.model import MultiClassLogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
import numpy as np
import re
import pickle as pk
from newspaper import Article
from train import Server_hidden
import threading

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/headline_prediction', methods=['POST'])




def headline_prediction():
    def getTitle(title):
        article = Article(title)
        article.download()
        article.parse()
        return article.title



    if(request.method == "POST"):
        tfidf = TfidfVectorizer()
        with open('tfidf.pk','rb') as fin:
            tfidf = pk.load(fin)

        y=np.genfromtxt("category.csv",dtype=int)

        log_reg = MultiClassLogisticRegression(y)
        title = request.get_json()[0]
        if(request.get_json()[1]=="Article Link"):
            title = getTitle(title)

        print(title)
        pred,prop=log_reg.predict_cat(title,tfidf)
      
        data = { "prediction":pred, "probability":round((prop*100),3), "title": title}
        return jsonify(data)


@app.route('/Add_new_data', methods=['POST'])
def Add_new_data():
    N=10000

    if(request.method == "POST"):
        cat_names = { 'Business and Politics': 'b' ,  'Science and Technology' : 't', 'Entertainment' :'e', 'Health' : 'h'}
        data= np.genfromtxt("new_data.csv", dtype=str, delimiter=',', skip_header=True)
        map={}
        if data.ndim > 1:
            for i in data[0:, 0]:
                map[i]=1
        elif data.shape[0]>0:
            map[data[0]]=1
        
        if request.get_json()[0] not in map.keys():
            data = np.array([[request.get_json()[0], cat_names[request.get_json()[1]]]])
            f = open("new_data.csv",'ab')
            np.savetxt(f, data, delimiter=',', fmt="%s")
            f.close()

            if (len(map.keys()) > N ):          #start trainining in the background
                sh = Server_hidden()
                p = threading.Thread(target=sh.train, args=[N])
                p.start()
        
        data={"Status": "completed"}
        return jsonify(data)




if __name__ == "__main__":
    app.run(debug=True)

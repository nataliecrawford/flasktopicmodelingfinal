#Written by Natalie Crawford
import json
import subprocess

from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS

from flask import send_from_directory
import re

import nltk
import matplotlib
import numpy as np
import pandas as pd
from pprint import pprint

# gensim
import gensim
import gensim.corpora as corpora
from Cython import inline
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.corpora.dictionary import Dictionary

# spacy for lemmatization
import spacy
from werkzeug.utils import secure_filename

spacy.cli.download("en_core_web_sm")

# plotting tools
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
#%matplotlib inline

# enable logging for gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# nltk stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.corpora.dictionary import Dictionary
from numpy import array
import IPython
import numpy as np
import logging
#import pyLDAvis.gensim
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from numpy import array
import pandas as pd
import io
import os


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")


import numpy as np
import logging

import pyLDAvis.gensim_models


from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.hdpmodel import HdpModel
from gensim.models.wrappers import LdaVowpalWabbit, LdaMallet
from gensim.corpora.dictionary import Dictionary
from numpy import array
import IPython
import numpy as np
import logging
import pyLDAvis.gensim_models
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from numpy import array
import pandas as pd
import io
'''
texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]


dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
'''


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#main page. file upload page
@app.route('/', methods=['GET', 'POST'])
def main():  # put application's code here
    return render_template('index.html')
    #requests.post(f"{server}/v1/cand", files={f"{target}_cand": cand_data})


# what the submit button sends to. Accesses data uploaded via saving the file to upload folder or "uploads"
# runs result function of the following route. if request is post then render the result html file
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
    global returndata
    returndata = topicmodeling(f.filename)
    result()
    if request.method == 'POST':
        return (render_template('result.html'))

# when called by the upload_file method, it checks if its a get method and if so then
# it returns the return data
@app.route('/result', methods=['GET'])
def result():
    local = returndata
    if request.method == 'GET':
        print(local)
        return jsonify(local)


#topic modeling method. sends the data from the uploaded file to the method
def topicmodeling(value):
    #written by amir karami and natalie crawford
    strvalue = str(value)
    strvalue=strvalue.replace(".txt", "")
    #access data for topic modeling
    p_df = io.open('/Users/nataliecrawford/Desktop/flasktopicmodeling/uploads/'+ value, encoding="latin-1")

    #terminal commands in order to find p(d/t)
    converttomallet = "/Users/nataliecrawford/Downloads/Mallet-202108/bin/mallet import-file --input /Users/nataliecrawford/Desktop/flasktopicmodeling/uploads/"\
                      + strvalue+".txt --output /Users/nataliecrawford/Downloads/" + strvalue +".mallet --keep-sequence --remove-stopwords"
    os.system(converttomallet)

    pofdocstotopicsandtopics = "/Users/nataliecrawford/Downloads/Mallet-202108/bin/mallet train-topics --input /Users/nataliecrawford/Downloads/"+ strvalue + ".mallet \
                    --num-topics 10 --num-iterations 1000 --num-top-words 10 --output-doc-topics /Users/nataliecrawford/Downloads/"+ strvalue+"_doctops.txt --output-topic-keys /Users/nataliecrawford/Downloads/"+strvalue+"_topics.txt"
    os.system(pofdocstotopicsandtopics)


    #topic weights
    exceldata = pd.read_table("/Users/nataliecrawford/Downloads/"+ strvalue+"_doctops.txt", header=None)
    df = pd.DataFrame(exceldata)

    # adds columns to easy access data
    x = 0
    y = 1
    numofcol = len(df.columns)
    columnnames = []
    while numofcol > 1:
        if x == 0:
            columnnames.append("docs")
            x = x + 1
        if x == 1:
            columnnames.append("year")
            x = x + 1
        else:
            columnnames.append("topic " + str(y))
            y = y + 1
        numofcol = numofcol - 1
    df.columns = columnnames
    numofrows = len(df)

    #sum vertically
    df = df.sum(axis=0)
    print(df)
    n = 2
    weights = df.values.tolist()
    # removes the first two columns in the sheet that are not the topics but are documents and the years
    weights = weights[n:]
    m = 1
    weightlist = []
    # total weight per topic
    topicweights = []
    for w in weights:
        # string array with the corresponding topic numbers to weights
        topicweights.append("Topic " + str(m) + "'s Weight is: " + str(round(w / numofrows, 3)))
        #rounds weights
        weightlist.append(str(round(w / numofrows, 3)))
        m = m + 1
    print(topicweights)





    #topics
    topics = pd.read_table("/Users/nataliecrawford/Downloads/"+strvalue+"_topics.txt", header=None)
    #topics are found for all rows, only the 3rd (2nd index) column
    topicarr = np.array(topics.iloc[:,2])
    print(topicarr)
    df2 = pd.DataFrame(data= topicarr, columns=['A'])
    print(df2)
    numofrows2 = len(df2)

    topiclist = []
    topiclist1 = []

    # adds commas between words
    for t in topicarr:
        topiclist1.append(str(t).replace(" ", ", "))

    # removes the comma and space on the end of each line of words
    for t in topiclist1:
        topiclist.append(t[:-2])

    print(topiclist)



    # the following besides the return statement and creation of return data is for
    # the purpose of calculating coherence



    # Create sample of 10,000 reviews
    #p_df = p_df.sample(n = 10000)
    # Convert to array
    docs =p_df.readlines()
    print(len(docs))
    #docs=docs[1:100]
    # Define function for tokenize and lemmatizing
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer

    def docs_preprocessor(docs):
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            docs[idx] = remove_stopwords(docs[idx])  # Remove stopwords
            docs[idx] = strip_punctuation(docs[idx]) # Remove punctuations
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isdigit()] for doc in docs]

        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 3] for doc in docs]


        # Lemmatize all words in documents.
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

        return docs
    # Perform function on our document
    docs = docs_preprocessor(docs)
    #Create Biagram & Trigram Models
    #from gensim.models import Phrases
    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    #bigram = Phrases(docs, min_count=10)
    #trigram = Phrases(bigram[docs])
    '''
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
        for token in trigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    '''
    #Remove rare & common tokens
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=10, no_above=0.2)
    #Create dictionary and corpus required for Topic Modeling
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    print(corpus[:1])

    '''
    goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=3000, num_topics=2)
    badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=3000, num_topics=2)
    
    goodcm = CoherenceModel(model=goodLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    badcm = CoherenceModel(model=badLdaModel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    
    #print(goodcm)
    print(goodcm.get_coherence())
    
    print(badcm.get_coherence())
    
    
    
    #model2 = LdaMallet('/Users/amirkarami/Documents/Research/mallet-2.0.8/bin/mallet',corpus=corpus , num_topics=10, id2word=dictionary, iterations=1000)


    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model=LdaMallet('/Users/nataliecrawford/Downloads/Mallet-202108/bin/mallet',corpus=corpus , num_topics=num_topics, id2word=dictionary, iterations=100)
            model_list.append(model)
            #coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

            coherence_values.append(coherencemodel.get_coherence())
            print(coherence_values)

        return model_list, coherence_values



    ###




    start=2
    limit=10
    step=1
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=start, limit=limit, step=step)

    x = range(start, limit, step)
    print(*x)
    print(coherence_values)
    d= list(zip(list(x), coherence_values))
    print(d)

    np.savetxt("/Users/nataliecrawford/Downloads/saveddata.csv", np.matrix(d), delimiter=',')

    #cm_cv_10 = CoherenceModel(model=model1, texts=docs, coherence='c_v')
    #cm_umass_10 = CoherenceModel(model=model1, texts=docs, coherence='u_mass')

    #cm2 = CoherenceModel(model=model2, texts=texts, coherence='c_v')

    #print("10",cm1.get_coherence())


    #print(cm2.get_coherence())

    
    model1 = LdaMallet('/Users/amirkarami/Documents/Research/mallet-2.0.8/bin/mallet',corpus=corpus , num_topics=10, id2word=dictionary, iterations=1)
    #model2 = LdaMallet('/Users/amirkarami/Documents/Research/mallet-2.0.8/bin/mallet',corpus=corpus , num_topics=10, id2word=dictionary, iterations=1000)
    
    
    cm1 = CoherenceModel(model=model1, texts=docs, coherence='c_v')
    #cm2 = CoherenceModel(model=model2, texts=texts, coherence='c_v')
    
    print(cm1.get_coherence())
    #print(cm2.get_coherence())
    
    
    hm = HdpModel(corpus=corpus, id2word=dictionary)
    topics = []
    for topic_id, topic in hm.show_topics(num_topics=10, formatted=False):
        topic = [word for word, _ in topic]
        topics.append(topic)
    
    #print(topics[:2])
    
    cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    
    
    print(cm.get_coherence())
    '''
    #calculates coherence using this ldamallet model
    model = LdaMallet('/Users/nataliecrawford/Downloads/Mallet-202108/bin/mallet', corpus=corpus, num_topics=10,
                       id2word=dictionary, iterations=101)
    cm = CoherenceModel(model=model, texts=docs, dictionary=dictionary, coherence='c_v')
    coherence = str(cm.get_coherence())

    #set up return data
    return_data = {
        "topics": topiclist,
        "weights": weightlist,
        "coherence": coherence
    }


    return return_data


if __name__ == '__main__':
    app.run()

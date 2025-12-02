# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"C:\Users\AR ANSARI\NIT\AI\AI Proj\Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# Cleaning the texts
import re 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

corpus = []  

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review) 



# Creating the TFIDF model
#from sklearn.feature_extraction.text import TfidfVectorizer
#cv = TfidfVectorizer()

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray() 




y = dataset.iloc[:, 1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)





from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)

#from sklearn.linear_model import LogisticRegression
#classifier =  LogisticRegression()

#from sklearn.ensemble import RandomForestClassifier
#classifier =  RandomForestClassifier()

#from sklearn.svm import SVC
#classifier =  SVC()

#from sklearn.neighbors import KNeighborsClassifier
#classifier =  KNeighborsClassifier()

#from sklearn.naive_bayes import MultinomialNB
#classifier =  MultinomialNB()

classifier.fit(X_train, y_train) 





# Predicting the Test set results
y_pred = classifier.predict(X_test) 

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
  
bias = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance


import gradio as gr
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def predict_sentiment(review_text):
    # Text Cleaning
    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)

    # Vectorize
    review_vector = cv.transform([review]).toarray()

    # Predict
    pred = classifier.predict(review_vector)[0]

    return "👍 Positive Review" if pred == 1 else "👎 Negative Review"

# ---------- Frontend UI ----------
ui = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Enter Review"),
    outputs=gr.Textbox(label="Prediction"),
    title="Restaurant Review Sentiment Analyzer",
    description="Write any restaurant review and get Positive or Negative prediction."
)

ui.launch()





#===============================================
'''
CASE STUDY --> model is underfitted  & we got less accuracy 

1> Implementation of tfidf vectorization , lets check bias, variance, ac, auc, roc 
2> Impletemation of all classification algorihtm (logistic, knn, randomforest, decission tree, svm, xgboost,lgbm,nb) with bow & tfidf 
4> You can also reduce or increase test sample 
5> xgboost & lgbm as well
6> you can also try the model with stopword 


6> then please add more recores to train the data more records 
7> ac ,bias, varian - need to equal scale ( no overfit & not underfitt)

'''
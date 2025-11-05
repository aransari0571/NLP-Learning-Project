# BOW  (bag of words)
import nltk

paragraph ="""I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to"""
# cleaning the texts
import re # re libary use to reguler expression

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)


corpus = []
#creat the empty list name as crpuse becuase after cleaning the data corpue wilst will store this clean data 

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ',sentences[i] )
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set (stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)

# creat the bag of words 

# also we called as document matrix

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_bow = cv.fit_transform(corpus).toarray()

  
# creat tfidf algo
#from sklearn.feature_extraction.text import TfidfVectorizer 
#tf = TfidfVectorizer()
#x_tf = tf.fit_transform(corpus).toarray()


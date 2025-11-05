
import nltk

from gensim.models import Word2Vec

from nltk.corpus import stopwords


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


#text preprocessing the data 
text = re.sub(r'\[[0-9]*\]', '', paragraph)
text = re.sub(r'\s+', '', text)
text = text.lower()
text = re.sub(r'\d', '', text)
text = re.sub(r'\s+', '', text)

# prepreing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

#Training the word2vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

similar = model.wv('freedom')


#finding word vectors
vectors = model.wv['freedm']

#most similar word
similar = model.wv.most_similar('freedom')

similar = model.wv.most_similar('viktam')

similar = model.wv.most_similar('son')









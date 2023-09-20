import joblib
X_train = ["This was really awesome an awesome movie",
           "Great movie! Ilikes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "really Dissapointed by the movie"]
# X_test1 = ["it was ok and really dissapointment","ok movie, good","One time watch","Needs improvement","Ending story line is could have been better"]
X_test = ["it was ok and really dissapointment"]

y_train = ["positive","positive","positive","positive","negative","negative","negative"] # 1- Positive class, 0- negative class
# y_test1 = ["negative","positive","positive","negative","negative"]
# y_test = ["negative"]


from nltk.tokenize import RegexpTokenizer
# NLTK -> Tokenize -> RegexpTokenizer

# Stemming
# "Playing" -> "Play"
# "Working" -> "Work"

from nltk.stem.porter import PorterStemmer
# NLTK -> Stem -> Porter -> PorterStemmer

from nltk.corpus import stopwords
# NLTK -> Corpus -> stopwords


# Downloading the stopwords
import nltk
nltk.download('stopwords')
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
print(en_stopwords)
ps = PorterStemmer()
def getCleanedText(text):
  text = text.lower()

  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in en_stopwords]
  stemmed_tokens = [ps.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text

# # Input from the user

X_clean = [getCleanedText(i) for i in X_train]
xt_clean = [getCleanedText(i) for i in X_test]


# Data before cleaning
'''
X_train = ["This was awesome an awesome movie",
           "Great movie! Ilikes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "Dissapointed by the movie"]
'''


# # Vectorize


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
# "I am PyDev" -> "i am", "am Pydev"


X_vec = cv.fit_transform(X_clean).toarray()


print(cv.vocabulary_)

Xt_vect = cv.transform(xt_clean).toarray()
print(Xt_vect)



# from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# mn = MultinomialNB()
# mn = DecisionTreeClassifier()
mn = LogisticRegression(random_state=42)

mn.fit(X_vec, y_train)

sd_model = joblib.dump(mn,'sd_model.pkl')

# y_pred = mn.predict(Xt_vect)
# print(y_pred[0].upper())

# # Acuura
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_pred,y_test))
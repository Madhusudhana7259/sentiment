import streamlit as st
import joblib
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
p = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+")
z = {'have', 'where', 'will', "won't", 'herself', 'didn', 'it', 'don', "mustn't", 'being', 'if', "don't", 'he', 'doing', 'who', 'between', 'themselves', 'am', 'hasn', 'you', 'because', 'a', "aren't", 'does', 'over', "hasn't", 'our', 'through', 'again', 'ain', 'wasn', "it's", 'each', 'won', 'itself', 'there', 'hers', 'they', 'mightn', 'just', 'which', 'these', 'under', 'once', 'very', 'some', 'and', 'is', 'both', "couldn't", 'too', 'then', 'couldn', 've', 'yourself', 'those', 'myself', 'whom', 'while', 'below', 'her', 'yourselves', "you'd", 'in', 'for', 're', 'than', 'other', "you've", 'mustn', 'as', 'here', 'from', 'ma', 'until', 'shan', 'what', 'd', 'isn', 'was', 'only', 'same', 'further', 'how', 'few', 'so', "weren't", 'when', 'all', 'by', 'had', 'at', 'ours', 'why', 'off', "you'll", 'most', "mightn't", 'their', 'my', 'needn', 'll', "haven't", 'do', 'down', "shan't", 'during', 'o', 'the', 'with', "that'll", "needn't", 'own', "should've", 'did', 'hadn', 'but', 'me', 'y', 'doesn', 'any', 'not', "shouldn't", 'i', 'be', 'theirs', 'ourselves', 'now', 'him', 'haven', 'she', "wouldn't", 'about', 'against', 't', "she's", 'aren', "wasn't", 'no', 'm', 'up', 'this', 'been', 's', 'wouldn', 'has', "hadn't", "isn't", 'can', 'that', 'to', 'yours', 'after', 'were', 'shouldn', 'his', 'or', 'more', 'such', 'should', "didn't", 'are', 'we', 'them', 'having', 'your', "you're", 'an', 'into', 'out', 'its', 'above', 'himself', 'on', "doesn't", 'weren', 'of', 'before', 'nor'}

X_train = ["This was really awesome an awesome movie",
           "Great movie! Ilikes it a lot",
           "Happy Ending! Awesome Acting by hero",
           "loved it!",
           "Bad not upto the mark",
           "Could have been better",
           "really Dissapointed by the movie"]
y_train = ["positive","positive","positive","positive","negative","negative","negative"]
def getCleanedText(text):
  text = text.lower()

  # tokenizing
  tokens = tokenizer.tokenize(text)
  new_tokens = [token for token in tokens if token not in z]
  stemmed_tokens = [p.stem(tokens) for tokens in new_tokens]
  clean_text = " ".join(stemmed_tokens)
  return clean_text

X_clean = [getCleanedText(i) for i in X_train]
X_vec = cv.fit_transform(X_clean).toarray()

# x = '''
# <style>
# body {
# background-image:url("https://www.telstra.com.au/content/dam/shared-component-assets/tecom/articles/elevate-your-cx/customer-service-1200x675.png");
# background-size: cover;
# }
# </style>
# '''
st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://www.telstra.com.au/content/dam/shared-component-assets/tecom/articles/elevate-your-cx/customer-service-1200x675.png");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
st.write("SENTIMENTAL ANALYSIS")
a = joblib.load('sd_model.pkl')
na = st.text_input("Enter a Review ")
if st.button('Submit'):
    resul = na.title()
    xt_clean = [getCleanedText(i) for i in [resul]]
    Xt_vect = cv.transform(xt_clean).toarray()
    st.success(a.predict(Xt_vect)[0].upper())







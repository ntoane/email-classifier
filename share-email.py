# Imports
import pandas as pd
import nltk
import re
import string as string
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression

# A list of labelled emails
emails = [
    ['I shared your email', 'shared'],
    ['I just shared your address', 'shared'],
    ["I've sent your email address to my friend", 'shared'],
    ["I've shared your email", 'shared'],
    ['I already shared email', 'shared'],
    ["I've just shared your address", 'shared'],
    ['Okay I have shared the email', 'shared'],
    ['I have shared your email', 'shared'],
    ['I did share your email', 'shared'],
    ['I shared your contacts', 'shared'],
    ['I shared your digits', 'shared'],
    ['I shared your contact details', 'shared'],
    ['I shared your contact card', 'shared'],
    ['I shared the email with my friends', 'shared'],
    ['I have sent this email to my friends', 'shared'],
    ['The email has been shared with all my friends', 'shared'],
    ['Can I share your email address', 'share'],
    ['May I share your email', 'share'],
    ['Might I share your email', 'share'],
    ['Could we share your email address with my friends', 'share'],
    ['Can I share your email with my friend', 'share'],
    ['Can I send your email to my friend', 'share'],
    ['Can I give your contacts with my friend?', 'share'],
]

# Create a data frame from a list of emails
data = pd.DataFrame(emails, columns=['email', 'label'])

# Encode shared=1, share=0
encoder = LabelEncoder()
data['label'] = encoder.fit_transform(data['label'])

# PorterStemmer
ps = PorterStemmer()
# Function to transform input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

data['transformed_text'] = data['email'].apply(transform_text)


# Feature Extraction using Bag of Words model (CountVectorizer) to convert the cleaned text into numeric features
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=16)

X = tfidf.fit_transform(data['transformed_text']).toarray()
y = data['label'].values

# Split the dataset into train and test before feature extraction
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

# Train Logistic Regression model
lrc = LogisticRegression(solver='liblinear', penalty='l1')
lrc.fit(X_train,y_train)

y_pred = lrc.predict(X_test)
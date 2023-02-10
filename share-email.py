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

data = pd.DataFrame(emails, columns=['email', 'label'])
print(data)
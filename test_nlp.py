import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

text = "I am feeling very stressed today"
tokens = word_tokenize(text)

print(tokens)

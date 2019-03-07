from flask import Flask
from flask_restful import Resource
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.externals import joblib
from nltk.corpus import stopwords
import re

class Search(Resource):
    def __get__(self):
    	searched_books = get_search_results(query,self.books,self.train_tfidf)
    	search_books = searched_books.to_dict("records")
    	return search_books,200

    def __post__(self):
    	searched_books = get_search_results(query)
    	search_books = searched_books.to_dict("records")
    	return search_books,200



def get_started():
	stop_words = stopwords.words('english')
	books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
	english_stemmer = SnowballStemmer('english')
	analyzer = CountVectorizer().build_analyzer()
	count = CountVectorizer(analyzer = stemming)
	count_matrix = count.fit_transform(books['all_text'])
	tfidf_transformer = TfidfTransformer()
	train_tfidf = tfidf_transformer.fit_transform(count_matrix)
	return books, train_tfidf



def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))

def clean_text(text):
	stop_words = stopwords.words('english')
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def get_search_results(query,books,train_tfidf):
    query = clean_text(query)
    query_matrix = count.transform([query])
    query_tfidf = tfidf_transformer.transform(query_matrix)
    similarity_score = cosine_similarity(query_tfidf, train_tfidf)
    sorted_indexes = np.argsort(similarity_score).tolist()
    return books.iloc[sorted_indexes[0][-10:]]
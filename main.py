import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from flask import Flask
# from flask_restful import Api
from flask_restful import Resource, Api
# import sklearn
from sklearn.externals import joblib
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from search import Search
nltk.download('stopwords')

# class Stemming:
#     def __init__(self,arg):
#         self.arg = arg

#     def stemming(self,text):
#         return (english_stemmer.stem(w) for w in analyzer(text))
# class Initial_process(object):
#     """docstring for Initial_process"""
#     def __init__(self,arg):
#            self.arg = arg
#     def get_started(self):
#         stop_words = stopwords.words('english')
#         print("Stop done")
#         books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
#         print("PDF done")
#         english_stemmer = SnowballStemmer('english')
#         print("Stemmer")
#         analyzer = CountVectorizer().build_analyzer()
#         print("Analyser")
#         # step = Stemming("start")
#         count = CountVectorizer(analyzer=self.stemming)
#         count_matrix = count.fit_transform(books['all_text'])
#         tfidf_transformer = TfidfTransformer()
#         train_tfidf = tfidf_transformer.fit_transform(count_matrix)
#         return books,train_tfidf,count,tfidf_transformer  

def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))
stop_words = stopwords.words('english')
books = pd.read_csv("gs://dataset_models/final_dataset.csv")
books.fillna("",inplace=True)
english_stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

# step = Stemming("start")
# count = CountVectorizer(analyzer=stemming)
# count_matrix = count.fit_transform(books['all_text'])
# tfidf_transformer = TfidfTransformer()
# trained_tfidf = tfidf_transformer.fit_transform(count_matrix)

model_directory = 'gs://dataset_models'
count_file = f'{model_directory}/document_term_matrix.pkl'
tfidf_file = f'{model_directory}/tfidf.pkl'
trained_tfidf_file = f'{model_directory}/trained_tfidf.pkl'

tfidf = joblib.load(tfidf_file)
count = joblib.load(count_file)
trained_tfidf = joblib.load(trained_tfidf_file)

        # return books,train_tfidf,count,tfidf_transformer  
#     def clean_text(self,text):
#         stop_words = stopwords.words('english')
#         text = re.sub('[^a-z \s]', '', text.lower())
#         text = [w for w in text.split() if w not in set(stop_words)]
#         return ' '.join(text)

#     def get_search_results(self,query,books,train_tfidf):
#         query = clean_text(query)
#         query_matrix = count.transform([query])
#         query_tfidf = tfidf_transformer.transform(query_matrix)
#         similarity_score = cosine_similarity(query_tfidf, train_tfidf)
#         sorted_indexes = np.argsort(similarity_score).tolist()
#         return books.iloc[sorted_indexes[0][-10:]]

# def stemming(self,text):
#         return (english_stemmer.stem(w) for w in analyzer(text))


# books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
# count = joblib.load(count_file)   
# tfidf = joblib.load(tfidf_file)
# trained_tfidf = joblib.load(trained_tfidf_file)
class Search(Resource):
    def __init__(self,**kwargs):
        # self.count = kwargs["count"]
        self.tfidf = kwargs["tfidf"]
        self.trained_tfidf = kwargs["trained_tfidf"]
        self.books = kwargs["books"]
        self.count = kwargs["count"]

    def get(self,query):
        # searched_books = get_search_results(str(query),self.books,self.train_tfidf)
        # query = clean_text(query)
        query = " ".join(query.split("_"))
        stop_words = stopwords.words('english')
        # query = re.sub('[^a-z \s]', '', query.lower())
        query = [w for w in query.split("_") if w not in set(stop_words)]
        query = ' '.join(query)
        print(query,"stemmer")
        # english_stemmer = SnowballStemmer('english')
        # analyzer = CountVectorizer().build_analyzer()
        # ip = Initial_process("start")
        # count = CountVectorizer(analyzer=stemming)
        query_matrix = self.count.transform([query])
        # tfidf_transformer = TfidfTransformer()
        query_tfidf = self.tfidf.transform(query_matrix)
        similarity_score = cosine_similarity(query_tfidf, self.trained_tfidf)
        sorted_indexes = np.argsort(similarity_score).tolist()
        searched_books = self.books.iloc[sorted_indexes[0][-10:]]
        search_books = searched_books.to_dict("records")
        return search_books,200

    def post(self):
    	searched_books = get_search_results(query)
    	search_books = searched_books.to_dict("records")
    	return search_books,200

app = Flask(__name__)
api = Api(app)
api.add_resource(Search,"/search/<string:query>",resource_class_kwargs={"books":books,"trained_tfidf":trained_tfidf,"count":count,"tfidf":tfidf})

if __name__ == "__main__":
    # stemming("doggy heavenly")
    # start = Initial_process("start")
    # count = joblib.load(count_file)
    # books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
    # tfidf = joblib.load(tfidf_file)
    # trained_tfidf = joblib.load(trained_tfidf_file)
    # print(type(books),type(trained_tfidf))
    # start.get_started()

    app.run(debug=True)

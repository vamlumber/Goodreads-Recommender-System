
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from flask import Flask
import gcsfs
from google.cloud import storage
from flask_restful import Resource, Api
# import sklearn
from sklearn.externals import joblib
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from google.cloud import logging
from flask import jsonify
nltk.download('wordnet')
nltk.download('stopwords')
# from search import Search
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
# fs = gcsfs.GCSFileSystem(project='Recommender-System')

# english_stemmer = SnowballStemmer('english')
# analyzer = CountVectorizer().build_analyzer()

# def ste(text):
    # return (english_stemmer.stem(w) for w in analyzer(text))

# stem = ste
# with fs.open('dataset_models/document_term_matrix.pkl', 'rb') as c:
# 	count = joblib.load(c)
# c.close()
# stop_words = stopwords.words('english')
# fileobj = get_byte_fileobj('my-project', 'my-bucket', 'my-path')
# df = pd.read_csv(fileobj)
# books = pd.read_csv("gs://dataset_models/final_dataset.csv")
# with fs.open('dataset_models/final_dataset.csv') as f:
#     books = pd.read_csv(f)
# books.fillna("",inplace=True)
# books = pd.read_csv("gs://dataset_models/final_dataset.csv",dtype={"work_id":float,"image_url":object,"url":object,"publication_year":float,"publication_month":float,"publication_day":float,"num_pages":object,"publisher":object,"description":object,"similar_books":object,"average_rating":float,"isbn":object,"original_title":object,"author_id":object,"authors":object,"all_text":object},low_memory=False)
# count = CountVectorizer(analyzer=stemming)
# count_matrix = count.fit_transform(books['all_text'])
# tfidf_transformer = TfidfTransformer()
# train_tfidf = tfidf_transformer.fit_transform(count_matrix)
# count_matrix = count.fit_transform(books['all_text'])
# tfidf_transformer = TfidfTransformer()
# trained_tfidf = tfidf_transformer.fit_transform(count_matrix)

# model_directory = 'gs://dataset_models'
# count_file = f'{model_directory}/document_term_matrix.pkl'
# tfidf_file = f'{model_directory}/tfidf.pkl'
# trained_tfidf_file = f'{model_directory}/trained_tfidf.pkl'

# tfidf = joblib.load(tfidf_file)
# count = joblib.load(count_file)
# trained_tfidf = joblib.load(trained_tfidf_file)

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

class getStarted:
    def __init__(self,path="gs://dataset_models/final_dataset.csv",bucket_name="dataset_models"):
            self.books = pd.read_csv(path,dtype={"work_id":float,"image_url":object,"url":object,"publication_year":float,"publication_month":float,"publication_day":float,"num_pages":object,"publisher":object,"description":object,"similar_books":object,"average_rating":float,"isbn":object,"original_title":object,"author_id":object,"authors":object,"all_text":object},low_memory=False)
            #self.storage_client = storage.Client()
            #self.bucket = self.storage_client.get_bucket(bucket_name)
            self.cv = CountVectorizer()
            self.count_matrix = self.cv.fit_transform(self.books["all_text"])
            self.tfidf = TfidfTransformer()
            self.trained_tfidf = self.tfidf.fit_transform(self.count_matrix)
		
    def get_book(self):
        return self.books
    def get_tfidf(self):
		#blob = self.bucket.blob("tfidf.pkl")
		#download that file and name it 'local.joblib'
		#blob.download_to_filename("tfidf.pkl")
		#load that file from local file
        return self.tfidf
    def document_term_matrix(self):
		#blob = self.bucket.blob("document_term_matrix.pkl")
		#download that file and name it 'local.joblib'
		#blob.download_to_filename("document_term_matrix.pkl")
        return self.cv
    def get_trained_tfidf(self):
		#blob = self.bucket.blob("trained_tfidf.pkl")
		#download that file and name it 'local.joblib'
		#blob.download_to_filename("trained_tfidf.pkl")
        return self.trained_tfidf

		# def get_stopwords(self):
		# 	return stopwords.words('english')

		# def get_count(self):
		# 	return CountVectorizer()
		# self.df = pd.read_csv(path,dtype={"work_id":float,"image_url":object,"url":object,"publication_year":float,"publication_month":float,"publication_day":float,"num_pages":object,"publisher":object,"description":object,"similar_books":object,"average_rating":float,"isbn":object,"original_title":object,"author_id":object,"authors":object,"all_text":object},low_memory=False)
		# self.fsi = fs
		
		# with self.fsi.open('dataset_models/tfidf.pkl', 'rb') as tf:
		# 	self.tfidf = joblib.load(tf)
		# tf.close()

		# with self.fsi.open('dataset_models/trained_tfidf.pkl', 'rb') as ttf:
		# 	self.trained_tfidf = joblib.load(ttf)
		# ttf.close()


		


# 		# self.count_file = f'{model}/document_term_matrix.pkl'
# 		# self.tfidf_file = f'{model}/tfidf.pkl'
# 		# self.trained_tfidf_file = f'{model}/trained_tfidf.pkl'

# 		# self.tfidf = joblib.load(self.tfidf_file)
		
		

		

class Initial(Resource):
    def __init__(self,**kwargs):
        self.books = kwargs["books"]

    def get(self):
        init_book = self.books.head(10).to_dict("records")
        return init_book,200
		
		
# books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
# count = joblib.load(count_file)   
# tfidf = joblib.load(tfidf_file)
# trained_tfidf = joblib.load(trained_tfidf_file)
class Search(Resource):
    def __init__(self,**kwargs):
        self.tfidf = kwargs["tfidf"]
        self.trained_tfidf = kwargs["trained_tfidf"]
        self.books = kwargs["books"]
        #self.count = CountVectorizer()
        self.count = kwargs["document_term_matrix"]
        self.stopwords = stopwords.words('english')
        #self.document_term_matrix = kwargs["document_term_matrix"]
        self.lemmatizer = WordNetLemmatizer()
        app.logger.info("Search Resource Loaded")
        # self.client = logging.Client()
    	# self.logger = self.client.logger('App_details')
    	# self.logger.log_text('Search Resource Loaded')  # API call

    def get(self,query):
    	# self.logger.log_text("Entering try")
    	try:
    		#query = " ".join(query.split("_"))
    		# stop_words = stopwords.words('english')
    		# query = re.sub('[^a-z \s]', '', query.lower())
    		query = [self.lemmatizer.lemmatize(w) for w in query.split("_") if w not in set(self.stopwords)]
    		query = ' '.join(query)
    		# self.logger.log_text("Lement")
    		# english_stemmer = SnowballStemmer('english')
    		# analyzer = CountVectorizer().build_analyzer()
    		# ip = Initial_process("start")
    		# count = CountVectorizer()
    		query_matrix = self.count.transform([query])
    		# self.logger.liog_text("fit_transform")
    		# tfidf_transformer = TfidfTransformer()
    		query_tfidf = self.tfidf.transform(query_matrix)
    		# self.logger.log_text("TfidfTransformer")
    		similarity_score = cosine_similarity(query_tfidf, self.trained_tfidf)
    		# self.logger.log_text("Cosine")
    		sorted_indexes = np.argsort(similarity_score).tolist()
    		# self.logger.log_text("sort")
    		searched_books = self.books.iloc[sorted_indexes[0][-10:]]
    		search_books = searched_books.to_dict("records")
    		
    	except Exception as e:
    		return jsonify(e),404
    	return search_books,200

    # def post(self):
    # 	searched_books = get_search_results(query)
    # 	search_books = searched_books.to_dict("records")
    # 	return search_books,200

initial_start = getStarted(path="gs://dataset_models/final_dataset.csv",bucket_name="dataset_models")

app = Flask(__name__)
api = Api(app)
api.add_resource(Initial,"/",resource_class_kwargs={"books":initial_start.get_book()})
api.add_resource(Search,"/search/<string:query>",resource_class_kwargs={"books":initial_start.get_book(),"trained_tfidf":initial_start.get_trained_tfidf(),"tfidf":initial_start.get_tfidf(),"document_term_matrix":initial_start.document_term_matrix()})

# if __name__ == "__main__":
    # stemming("doggy heavenly")
    # start = Initial_process("start")
    # count = joblib.load(count_file)
    # books = pd.read_csv("/home/abhishek/Abe/Datasets/final_dataset.csv")
    # tfidf = joblib.load(tfidf_file)
    # trained_tfidf = joblib.load(trained_tfidf_file)
    # print(type(books),type(trained_tfidf))
    # start.get_started()

    # app.run(debug=False,port=8080,host='0.0.0.0')

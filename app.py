from flask import Flask
from flask_restful import Api
from ./Models/model.search import Search

app = Flask(__name__)
api = Api(app)

api.addResource(Search,"/search/<string:query>",resource_class_kwargs={'books': books,'train_tfidf':train_tfidf})

if __name__ == "__main__":
    # app,api = create_app("config")
    books , train_tfidf = get_started()
    app.run(debug=True)

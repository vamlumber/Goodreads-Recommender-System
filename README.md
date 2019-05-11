# Goodreads-Recommender-System

## Introduction
This  is a Rest Api Server , which provides recommendation for the books. I have used flask backend framework to create REST calls and deployed it on Google app engine. The dataset for the project has been stored in google cloud storage.
The 'app.yaml' file is my google app engine configure file . The 'requirement.txt' contains all the libraries ,
I have used in the project.Also, my kaggle note book is also add the repository , which I have used to perform my simulation , test and experimentation.

## Deployment
Pre-requistes for the project is python3 and pip. Run command 'pip install -r requirements.txt' to install all the necessary libraries to run the application. 
To run the file in the development server , you need run it by running command 'python main.py' by commandline once inside the folder. You will also need to specify the path of the dataset for the set by making changes in the 'main.py' file.

Note: Depending on your deployment platform , place the dataset and mention the path correctly.

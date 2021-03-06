# Sentiment-Analysis-in-German-Language
## Created a dockerized, simple REST API, with Python & FastAPI, to predict Sentiment in the German language
The REST API model is a simple sentiment analyzer, it have:

1. An endpoint to predict sentiment given a plain text.  
2. An endpoint to predict sentiment given a file with plain text.  
3. An endpoint to predict sentiment given a json file with a field containing plain text.

### The possible outcomes of the sentiment analysis are: {Positive, Neutral, Negative}.
* I used a pre-trained existing model which is trained on 1.834 million samples. The training data contains texts from various domains like Twitter, Facebook and movie, app and hotel reviews.  
* All endpoints are returning the prediction in the same json format. Moreover, the returned json also indicating the prediction's confidence probability.  
* Used Python Poetry for dependency management.  

# To run the application
- clone the repo 
```bash
git clone https://github.com/ashish-narwal/Sentiment-Analysis-in-German-Language
cd Sentiment-Analysis-in-German-Language
```

- Install ```Python 3.8``` and ```poetry``` 
- Run the command ```poetry shell``` to create a virtual environment
- Run ```poetry install``` to install the dependencies
# To run locally
- Run ```uvicorn src.main:app --reload``` to run the application on local host that we can access through http://127.0.0.1:8000
- All routes are available on ```http://127.0.0.1:8000/docs``` or ```http://127.0.0.1:8000/redoc``` paths with Swagger or ReDoc.

# To run with docker
- Make sure  ```Docker ``` is running locally.
- Run  ```docker build -t sentiment:latest . ``` to build the docker image with name sentiment
- Run  ```docker run -d -p 80:80 sentiment:latest``` to launch the container 
- Now we have a container runnning on our local machine that we access through  http://0.0.0.0:80

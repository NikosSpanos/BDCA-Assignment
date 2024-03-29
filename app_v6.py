# Big Data & Content Analytics Project

# Students FullName: Spanos Nikolaos, Baratsas Sotirios
# Students ID: f2821826, f2821803 
# Supervisor: Papageorgious Xaris, Perakis Georgios

# This is the first of the two components for the movie chatbot to run efficiently. In this python file the reader will find the main functions for the webhook responses to work.


# import flask dependencies

from flask import Flask, request, make_response, jsonify
from movie_recommendation_v5 import recommend_movie
import sys

# initialize the flask app
app = Flask(__name__)

# default route
@app.route('/')
def index():
    return 'Hello World!'

res = None

# Function 1

def suggest_movie(req):

    global res

    # fetch action from json
    user_genre = req.get('queryResult').get('parameters').get('user_genre')

    user_prefrences = req.get('queryResult').get('parameters').get('user_prefrences')
    
    user_movie = req.get('queryResult').get('parameters').get('user_movie')

    res = recommend_movie(user_genre, user_prefrences, user_movie)

    return {'fulfillmentText': "Thank you! Based on your preferences, I propose the following two movies:\n\n{0} \n(IMDB Rating: {1} & Link: {2}) \n\n{3} \n(IMDB Rating: {4} & Link: {5})\n\nHave you seen any of these movies?".format(res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2])}

# Function 2

def propose_second_movie(req):

    ask_seen = req.get('queryResult').get('parameters').get('ask_seen_one')

    if ask_seen == 'yes' or ask_seen == 'Yes':
        
        return {'fulfillmentText': 'Then I propose you to see this movie: \n\n{0} \n(IMDB Rating: {1} & Link: {2})\n\n(Type "Thank you" if would like to end the conversation!)'.format(res[2][0], res[2][1], res[2][2])}

    elif ask_seen == 'no' or ask_seen == 'No':
        
        return {'fulfillmentText': 'Great news! Grab a bowl of Pop-Corn and enjoy your film!\n\n(Type "Thank you" if would like to end the conversation!)'}

# Function 3

def thanks(req):

    greeding = req.get('queryResult').get('parameters').get('greeding')

    if greeding == 'Thank you' or greeding == 'thank you' or greeding == 'Thanks' or greeding == 'thanks' or greeding == 'Nice' or greeding == 'nice':

        return {'fulfillmentText': "You're welcome! :)"}


# Create a route for webhook 

# For webhook to properly work, I have to copy paste the https link from ngrok

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():

    # return response
    req = request.get_json(force=True, silent = True)

    action = req.get('queryResult').get('action')

    intent_name = req.get('queryResult').get('intent').get('displayName')

    if action == "get_results" and intent_name == 'KellyMovieBot':

        return make_response(jsonify(suggest_movie(req)))

    elif action == "ask_question" and intent_name == 'ask_first_time':

        return make_response(jsonify(propose_second_movie(req)))

    elif action == "thanks_giving_end" and intent_name == "thanks_giving":

        return make_response(jsonify(thanks(req)))

# run the app
if __name__ == '__main__':
   app.run()
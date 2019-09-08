# Big Data & Content Analytics Project

# Students FullName: Spanos Nikolaos, Baratsas Sotirios
# Students ID: f2821826, f2821803 
# Supervisor: Papageorgious Xaris, Perakis Georgios

# This is the second of the two components for the messenger chatbot to word. It is actually the 6th version of the movie recommendation algorithm. 


# Code snipset that wiil provide the response

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import pickle

# -----------------------Functions used ----------------------------------------------------------

def get_index_from_input_movie(user_input):
    return five_thousands[five_thousands.movie_title.str.lower().str.replace('-', '').replace('The', '').replace(':', '') == user_input]['index'].values[0]
    
def stop_and_stem(uncleaned_list):
    ps = PorterStemmer()
    stop = set(stopwords.words('english'))
    stopped_list = [i for i in uncleaned_list if i not in stop]
    stemmed_words = [ps.stem(word) for word in stopped_list]
    return stemmed_words

def search_words(row, list_of_words):
    ps = PorterStemmer()
    row = [ps.stem(x) for x in row]
    counter = 0
    for word in list_of_words:
        if word in row:
            counter = counter + 1
    return counter

def find_correct_genre(user_input, genre_list):
    scores_sim=[]
    vectorizer = TfidfVectorizer()

    for item in genre_list:
        ed = nltk.edit_distance(user_input, item)
        scores_sim.append(ed)
    correct_genre_index = scores_sim.index(min(scores_sim))
    correct_genre = genre_list[correct_genre_index].lower()
    print(correct_genre)
    return correct_genre

def find_correct_movie(user_input, movie_list):
    scores_similarity=[]

    for item in movie_list:
        ed = nltk.edit_distance(user_input, item)
        scores_similarity.append(ed)
    correct_movie_index = scores_similarity.index(min(scores_similarity))
    correct_movie = movie_list[correct_movie_index].lower()
    return correct_movie


# -----------------------------------------------------------------------------------------------


# Import the dataset

five_thousands = pd.read_pickle('C:\\Users\\dq186sy\\Desktop\\Big Data Content Analytics\\Movie Recommendation System\\five_thousands_embedded_02092019.pkl')

five_thousands = five_thousands.drop(['level_0', 'index'], axis = 1)

five_thousands = five_thousands.reset_index()

five_thousands['index'] = np.arange(0, len(five_thousands))


# -------------------------------------------------------------------------------------------------

def recommend_movie(input_one, input_two, input_movie):

    # Create the movie_genre list with the unique types of genre 

    movie_genre_first = five_thousands.genre_0.unique().tolist()
    movie_genre_second = five_thousands.genre_1.unique().tolist()
    movie_genre_third = five_thousands.genre_2.unique().tolist()
    movie_genre_fourth = five_thousands.genre_3.unique().tolist()
    movie_genre_fifth = five_thousands.genre_4.unique().tolist()
    movie_genre_sixth = five_thousands.genre_5.unique().tolist()
    movie_genre_seventh = five_thousands.genre_6.unique().tolist()
    movie_genre_eight = five_thousands.genre_7.unique().tolist()

    movie_genre_list = np.asarray(movie_genre_first + movie_genre_second + movie_genre_third + movie_genre_fourth + movie_genre_fifth + movie_genre_sixth + movie_genre_seventh + movie_genre_eight)
    list(movie_genre_list.flatten())
    movie_genre_list = list(set(movie_genre_list))

    movie_genre_list = [x.lower() for x in movie_genre_list]


    # -------------------------------------------------------------------------------------------------


    # Get inputs from the user and clean them

    input_one = find_correct_genre(input_one.lower(), movie_genre_list)

    input_two = input_two.lower().replace(',', '').replace('.', '').split(' ')

    inputs_list = stop_and_stem(input_two)

    input_movie = input_movie.lower().replace('-', '').replace('The', '').replace(':', '')


    # -------------------------------------------------------------------------------------------------


    # Using the genre input given by the user, isolate those movies that match the given genre (i.e Action movies)

    locked_frame = five_thousands.loc[(five_thousands.genre_0.str.lower() == input_one) | (five_thousands.genre_1.str.lower() == input_one) | (five_thousands.genre_2.str.lower() == input_one) | (five_thousands.genre_3.str.lower() == input_one) | (five_thousands.genre_4.str.lower() == input_one) | (five_thousands.genre_5.str.lower() == input_one) | (five_thousands.genre_6.str.lower() == input_one) | (five_thousands.genre_7.str.lower() == input_one)]

    indexes_list = locked_frame.index.tolist()

    locked_frame['index'] = np.arange(0, len(locked_frame))


    # -------------------------------------------------------------------------------------------------

    # Check of the movie user gave is in the movie list of the dataset

    with open('C:\\Users\\dq186sy\\Desktop\\Big Data Content Analytics\\Movie Recommendation System\\movie_title_list.pkl', 'rb') as f:
        movies_list = pickle.load(f)

    if input_movie in movies_list:

        input_movie = find_correct_movie(input_movie, movies_list)
        
        # Isolate the movie plot of the movie provided from the user [If the movie is part of the dataset].

        movie_plot_new = locked_frame['plot_summary'].loc[(locked_frame.movie_title.str.lower().str.replace('-', '').str.replace('The', '').str.replace(':', '') == input_movie)].apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower())))).values[0]

        cleaned_movie_plot = stop_and_stem(movie_plot_new)

        plot_user_input_list = inputs_list + cleaned_movie_plot


        # -------------------------------------------------------------------------------------------------


        # Get the index of the movie provied by the user

        movie_index = get_index_from_input_movie(input_movie)

        # -------------------------------------------------------------------------------------------------


        # Get Features Embeddings based on the movie_index

        feature_vector = five_thousands['average_combined_features'][five_thousands['index'] == movie_index]


        # Get the Embeddings of the movies matched the user's genre (i.e of all the action movies)

        with open('C:\\Users\\dq186sy\\Desktop\\Big Data Content Analytics\\Movie Recommendation System\\my_embeddings_array_updated_02092019.pkl', 'rb') as f:
            my_embeddings_array_updated = pickle.load(f)

        genre_embeddings_array = my_embeddings_array_updated[indexes_list]


        # -------------------------------------------------------------------------------------


        # Concatenate the embeddings of the combined features

        selected_movie_vector = np.hstack([feature_vector.apply(pd.Series).values])

        # Calculate Cosine Distance

        cosine_dist = cosine_distances(genre_embeddings_array, selected_movie_vector.reshape(1,-1))

        # Get the similar movies & Slice the dataframe on the top 5 most similar movies to the movie given  by the user

        movie_return = np.argsort(cosine_dist, axis=None).tolist()[1:6]

        locked_frame_new = locked_frame[locked_frame['index'].isin(movie_return)]


        # -------------------------------------------------------------------------------------


        # Create two new columns "Unique Words" + "Number of words"

        # Create the new column of "UNIQUE" words of the combined features
        locked_frame_new['unique_words'] = locked_frame_new.combined_features.apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower()))))

        # Create the column "Number of words" for each word contained in the unique words column
        locked_frame_new['number_of_words'] = locked_frame_new.unique_words.apply(search_words, args=(plot_user_input_list,))


        # -------------------------------------------------------------------------------------


        # Calculate the movie score

        primary_genre = list([(locked_frame.genre_0.str.lower() == input_one)*0.1, (locked_frame.genre_1.str.lower() == input_one)*0.1])

        locked_frame_new['movie_score'] = 0.3*locked_frame_new.updated_rating.astype(float) + 0.5*locked_frame_new.number_of_words

        locked_frame_new['movie_score'] = locked_frame_new['movie_score'] + primary_genre[0] + primary_genre[1]


        # -------------------------------------------------------------------------------------


        # Give to the user the proper movie recommendation

        top_three_rows = locked_frame_new.nlargest(3, 'movie_score')
        
        top_three_rows.rename(columns={'movie_title':'Movie Title', 'updated_rating':'IMDB Rate', 'movie_imdb_link':"Movie's Link"}, inplace=True)

        # Recommend the movie

        recommendations_list = top_three_rows.loc[:, ['Movie Title', 'IMDB Rate', "Movie's Link"]].values.tolist()
        
        return recommendations_list
        
    else:
        
        plot_user_input_list = inputs_list
        
        locked_frame['unique_words'] = locked_frame.combined_features.apply(lambda x: list(set(re.split(' |,|\n', x.strip().lower()))))

        locked_frame['number_of_words'] = locked_frame.unique_words.apply(search_words, args=(plot_user_input_list,))
        
        primary_genre = list([(locked_frame.genre_0.str.lower() == input_one)*0.1, (locked_frame.genre_1.str.lower() == input_one)*0.1])

        locked_frame['movie_score'] = 0.3*locked_frame.updated_rating.astype(float) + 0.5*locked_frame.number_of_words
        
        locked_frame['movie_score'] = locked_frame['movie_score'] + primary_genre[0] + primary_genre[1]
        
        
        # Give to the user the proper movie recommendation

        top_three_rows = locked_frame.nlargest(3, 'movie_score')
        
        top_three_rows.rename(columns={'movie_title':'Movie Title', 'updated_rating':'IMDB Rate', 'movie_imdb_link':"Movie's Link"}, inplace=True)

        
        # Recommend the movie

        recommendations_list = top_three_rows.loc[:, ['Movie Title', 'IMDB Rate', "Movie's Link"]].values.tolist()
        
        return recommendations_list
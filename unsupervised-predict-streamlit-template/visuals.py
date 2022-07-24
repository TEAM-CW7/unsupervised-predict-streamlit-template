import pandas as pd
import numpy as np
import scipy as sp 

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
imdb_data = pd.read_csv('resources/data/imdb_data.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
# eda_df = train_df[train_df['userId']!=72315]
ratings_df.drop(['timestamp'], axis=1,inplace=True)

def feature_frequency(df, column):
    """
    Function to count the number of occurences of metadata such as genre
    Parameters
    ----------
        df (DataFrame): input DataFrame containing movie metadata
        column (str): target column to extract features from
    Returns
    -------
        
    """
    # Creat a dict to store values
    df = df.dropna(axis=0)
    genre_dict = {f'{column}': list(),
                 'count': list(),}
    # Retrieve a list of all possible genres
    print('retrieving features...')
    for movie in range(len(df)):
        gens = df[f'{column}'].iloc[movie].split('|')
        for gen in gens:
            if gen not in genre_dict[f'{column}']:
                genre_dict[f'{column}'].append(gen)
    # count the number of occurences of each genre
    print('counting...')
    for genre in genre_dict[f'{column}']:
        count = 0
        for movie in range(len(df)):
            gens = df[f'{column}'].iloc[movie].split('|')
            if genre in gens:
                count += 1
        genre_dict['count'].append(count)
        
        # Calculate metrics
    data = pd.DataFrame(genre_dict)
    print('done!')
    return data

def feature_count(df, column):
    fig= plt.figure(figsize=(10,6))
    ax = sns.barplot(y = df[f'{column}'], x = df['count'], palette='cool', orient='h')
    plt.title(f'Number of Movies per {column}', fontsize=14)
    plt.ylabel(f'{column}')
    plt.xlabel('Count')
    st.pyplot(fig)

# def mean_calc(feat_df, ratings = eda_df, movies = movies_df, metadata = imdb_df, column = 'genres'):
    
    
    
#     mean_ratings = pd.DataFrame(ratings.join(movies, on='movieId', how='left').groupby(['movieId'])['rating'].mean())
#     movie_eda = movies.copy()
#     movie_eda = movie_eda.join(mean_ratings, on = 'movieId', how = 'left')

#     # Exclude missing values
#     movie_eda = movie_eda
#     movie_eda2 = movie_eda[movie_eda['rating'].notnull()]

#     means = []
#     for feat in feat_df[f'{column}']:
#         mean = round(movie_eda2[movie_eda2[f'{column}'].str.contains(feat)]['rating'].mean(),2)
#         means.append(mean)
#     return means

# def genre_popularity(df):
#     """
#     Plots the mean rating per genre.
#     """
#     count_filt = 500
#     plt.figure(figsize=(10,6))
#     plot_data = df[df['count']>count_filt]
#     mean = plot_data['mean_rating'].mean()
#     min_ = plot_data['mean_rating'].min()
#     max_ = plot_data['mean_rating'].max()
#     sns.barplot(y = plot_data['genres'], x = plot_data['mean_rating'], order = plot_data['genres'], orient='h',palette='cool')
#     plt.axvline(x=mean, label = f'mean {round(mean,1)}' , color='black', lw=1, ls ='--')
#     plt.axvline(x=min_, label = f'min {round(min_,1)}' , color='#4D17A0', lw=1, ls = '--')
#     plt.axvline(x=max_, label = f'max {max_}' , color='#4DA017', lw=1,ls = '--')
#     plt.title(f'Mean Rating Per Genre', fontsize=14)
#     plt.ylabel('Genre')
#     plt.xlabel('Mean Rating')
#     plt.legend(loc='lower center')
#     plt.show()

def plot_eda():
    genres = feature_frequency(movies_df, 'genres')
    feature_count(genres.sort_values(by = 'count', ascending=False), 'genres')   
    # genres['mean_rating'] = mean_calc(genres) 
    # genre_popularity(genres.sort_values('mean_rating', ascending=False))
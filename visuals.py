import pandas as pd
import numpy as np
import scipy as sp 

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
imdb_df = pd.read_csv('resources/data/imdb_data.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
# train_df = pd.read_csv('resources/data/train.csv')

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

def mean_calc(feat_df, ratings = ratings_df, movies = movies_df, metadata = imdb_df, column = 'genres'):
    
    
    
    mean_ratings = pd.DataFrame(ratings.merge(movies, on='movieId', how='left').groupby(['movieId'])['rating'].mean())
    movie_eda = movies.copy()
    movie_eda = movie_eda.join(mean_ratings, on = 'movieId', how = 'left')

    # Exclude missing values
    movie_eda = movie_eda
    movie_eda2 = movie_eda[movie_eda['rating'].notnull()]

    means = []
    for feat in feat_df[f'{column}']:
        mean = round(movie_eda2[movie_eda2[f'{column}'].str.contains(feat)]['rating'].mean(),2)
        means.append(mean)
    return means

def genre_popularity(df):
    """
    Plots the mean rating per genre.
    """
    count_filt = 500
    fig = plt.figure(figsize=(10,6))
    plot_data = df[df['count']>count_filt]
    mean = plot_data['mean_rating'].mean()
    min_ = plot_data['mean_rating'].min()
    max_ = plot_data['mean_rating'].max()
    sns.barplot(y = plot_data['genres'], x = plot_data['mean_rating'], order = plot_data['genres'], orient='h',palette='cool')
    plt.axvline(x=mean, label = f'mean {round(mean,1)}' , color='black', lw=1, ls ='--')
    plt.axvline(x=min_, label = f'min {round(min_,1)}' , color='#4D17A0', lw=1, ls = '--')
    plt.axvline(x=max_, label = f'max {max_}' , color='#4DA017', lw=1,ls = '--')
    plt.title(f'Mean Rating Per Genre', fontsize=14)
    plt.ylabel('Genre')
    plt.xlabel('Mean Rating')
    plt.legend(loc='lower center')
    st.pyplot(fig)

def ratings_distplot(df, column='rating'):
    """
    Plots the distribution of ratings in the dataset.
    Parameters
    ----------
        df (DataFrame): input DataFrame
        column (str): column to plot
    Returns
    -------
        distplot (NoneType): distplot of rating frequencies
    """
    fig = plt.figure(figsize=(8,6))
    ax = sns.distplot(df[f'{column}'],bins=10, kde=False, hist_kws=dict(alpha=0.6),color="#4D17A0")
    mean = df[f'{column}'].mean()
    median = df[f'{column}'].median()
    plt.axvline(x=mean, label = f'mean {round(mean,2)}' , color='#4D17A0', lw=3, ls = '--')
    plt.axvline(x=median, label = f'median {median}' , color='#4DA017', lw=3, ls = '--')
    plt.xlim((0.5,5))
    plt.ylim((0,2500000))
    plt.title(f'Distribution of Ratings', fontsize=14)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot(fig) 

def mean_ratings_scatter(df, color='#F33DD8', column='userId'):
    """
    Make scatterplots of mean ratings.
    Parameters
    ----------
        df (DataFrame): input DataFrame
        color (str): plot colour
        column (str): column to plot
    Returns
    -------
        scatterplot (NoneType): scatterplot of mean number of ratings
    """
    fig =plt.figure(figsize=(6,4))
    mean_ratings = df.groupby(f'{column}')['rating'].mean()
    user_counts = df.groupby(f'{column}')['movieId'].count().values
    sns.scatterplot(x=mean_ratings, y = user_counts, color=color)
    plt.title(f'Mean Ratings by Number of Ratings', fontsize=14)
    plt.xlabel('Rating')
    plt.ylabel('Number of Ratings')
    st.pyplot(fig) 

# Function that counts the number of times each of the genre keywords appear
def count_word(dataset, ref_col, census):
   
    keyword_count = dict()
    for s in census: 
        keyword_count[s] = 0
    for census_keywords in dataset[ref_col].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]: 
            if pd.notnull(s): 
                keyword_count[s] += 1
    
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)

    return keyword_occurences, keyword_count


def wordClouds():
    sys = st.radio("Select a feature",
                        ('Titles',
                            'Cast',
                            'Genre'))
    if sys == 'Titles':
        st.info('This WordCloud displays the most popular movie titles')
        # Creating a wordcloud of the movie titles to view the most popular movie titles within the word cloud
        movies_df['title'] = movies_df['title'].fillna("").astype('str')
        title_corpus = ' '.join(movies_df['title'])
        title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)

        # Plotting the wordcloud
        fig = plt.figure(figsize=(16,8))
        plt.imshow(title_wordcloud)
        plt.axis('off')
        st.pyplot(fig)

    if sys == 'Cast':
        st.info('This WordCloud displays the most popular movie title_cast ')
    # Creating a wordcloud of the movie titles to view the most popular movie title_cast within the word cloud
        imdb_df['title_cast'] = imdb_df['title_cast'].fillna("").astype('str')
        title_corpus = ' '.join(imdb_df['title_cast'])
        title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=2000, width=4000).generate(title_corpus)

        # Plotting the wordcloud
        fig = plt.figure(figsize=(16,8))
        plt.imshow(title_wordcloud)
        plt.axis('off')
        st.pyplot(fig)

    if sys == 'Genre':
        genre_labels = set()
        for s in movies_df['genres'].str.split('|').values:
            genre_labels = genre_labels.union(set(s))
        
        keyword_occurences, dum = count_word(movies_df, 'genres', genre_labels)
     

    # Define the dictionary used to produce the genre wordcloud
        genres = dict()
        trunc_occurences = keyword_occurences[0:18]
        for s in trunc_occurences:
            genres[s[0]] = s[1]

        # Create the wordcloud
        genre_wordcloud = WordCloud(width=4000,height=2000, background_color='white')
        genre_wordcloud.generate_from_frequencies(genres)

        # Plot the wordcloud
        f, ax = plt.subplots(figsize=(16, 8))
        plt.imshow(genre_wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()
        st.pyplot(f)


def plot_eda():
    options= ["Genre Counts", "Movie Ratings", "Popular Words per Feature"]
    visuals_selection = st.sidebar.selectbox("Select Visual", options)
    if visuals_selection == "Genre Counts":
        genres = feature_frequency(movies_df, 'genres')
        feature_count(genres.sort_values(by = 'count', ascending=False), 'genres')   
    if visuals_selection == "Movie Ratings": 
        genres = feature_frequency(movies_df, 'genres')   
        genres['mean_rating'] = mean_calc(genres) 
        genre_popularity(genres.sort_values('mean_rating', ascending=False))
    if visuals_selection == "Popular Words per Feature":
        wordClouds()
    
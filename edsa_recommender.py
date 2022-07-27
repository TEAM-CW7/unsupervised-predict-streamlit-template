"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from calendar import c
from turtle import width
import streamlit as st

import streamlit.components.v1 as components
# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from visuals import plot_eda
from carousel import carousel
import hydralit_components as hc
import hydralit as hy
import streamlit_modal as modal


# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')



# App declaration
app = hy.HydraApp(title='Pegasus AI  Movie Recommender',favicon="🐙",hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True)


@app.addapp()
def About ():

    st.write('Pegasus AI  Movie Recommender')
    st.video('https://www.youtube.com/watch?v=Qjq9rm-DvkI')

@app.addapp()
def Recommender():

        # DO NOT REMOVE the 'Recommender System' option below, however,
        # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Help"]

        # -------------------------------------------------------------------
        # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
        # -------------------------------------------------------------------
        # page_selection = st.sidebar.selectbox("Choose Option", page_options)
        # if page_selection == "Recommender System":
            # Header contents
    st.write('# Movie Recommender Engine')
    st.write('### EXPLORE Data Science Academy Unsupervised Predict')
    carousel()
        # st.image('resour0ces/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
    sys = st.radio("Select an algorithm",
                        ('Content Based Filtering',
                            'Collaborative Based Filtering'))

            # User-based preferences
    st.write('### Enter Your Three Favorite Movies')
    movie_1 = st.selectbox('First Option',title_list[14930:15200])
    movie_2 = st.selectbox('Second Option',title_list[25055:25255])
    movie_3 = st.selectbox('Third Option',title_list[21100:21200])
    fav_movies = [movie_1,movie_2,movie_3]

            # Perform top-10 movie recommendation generation
    if sys == 'Content Based Filtering':
        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                                top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                                We'll need to fix it!")


    if sys == 'Collaborative Based Filtering':
        if st.button("Recommend"):
            try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = collab_model(movie_list=fav_movies,
                                                            top_n=10)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    st.subheader(str(i+1)+'. '+j)
            except:
                st.error("Oops! Looks like this algorithm does't work.\
                                    We'll need to fix it!")
    # modal1()
    modal2()

def modal1():    
    open_modal1 = st.button("What is a content based recommendation?")
    if open_modal1:
        modal.open()

    if modal.is_open():
        with modal.container():
            st.markdown("**<font color=‘blue’>Content Based Recommender</font>**", unsafe_allow_html=True)


            html_string = '''
        <h1>HTML string in RED</h1>

        <script language="javascript">
          document.querySelector("h1").style.color = "red";
        </script>
        '''
            components.html(html_string)

            st.write("")
            value = st.checkbox("Check me")
            st.write(f"Checkbox checked: {value}")


def modal2():
                
    open_modal2 = st.button("What is a collaborative  based recommendation?")
    if open_modal2:
        modal.open()

    if modal.is_open():
        with modal.container():
            st.markdown(
    """
    <style>
    .reportview-container {
        background: url("url_goes_here")
    }
   .sidebar .sidebar-content {
        background: url("resources/imgs/pegasus.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)
            # new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Collaborative Based Recommendation</p>'
            # st.markdown(new_title, unsafe_allow_html=True)
            # st.markdown("<font color=‘blue’>Collaborative Based Recommendation</font>", unsafe_allow_html=True ,  font-size= 42px;)
            html_string = '''
        <h1>Collaborative Based Recommendation</h1>

        <script language="javascript">
          document.querySelector("h1").style.color = "red";
        </script>
        '''
            
            components.html(html_string)
            st.markdown("<font color= 'red'>Find out more more [here](https://www.youtube.com/watch?v=h9gpufJFF-0)</font>",unsafe_allow_html=True)
        
                                             


@app.addapp()
def Dashboard():
    plot_eda()


@app.addapp()    
def Help():
    st.write("Want Help?")
    
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
    
        st.header("Content Based")
        st.video("https://www.youtube.com/watch?v=MHL0ImqqeJ8", width=200)
        st.write("Read [more] (https://developers.google.com/machine-learning/recommendation/content-based/basics)")

    with col2:
        st.header("Collaborative")
        st.video("https://www.youtube.com/watch?v=h9gpufJFF-0", width=200)





@app.addapp()
def ContactUs():
    st.header(":mailbox: Get In Touch With Us !")


    contact_form = """
<form action="https://formsubmit.co/morglin.olivierm@gmail.com" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
"""

    st.markdown(contact_form, unsafe_allow_html=True)

    local_css("resources/style/style.css")  

# Use Local CSS File
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

   
app.run()
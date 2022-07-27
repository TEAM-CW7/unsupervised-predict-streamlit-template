import streamlit as st

import streamlit.components.v1 as components


def carousel():

    imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

    imageUrls = [
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/DarKnight.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/GodFather.jpeg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/SchindlersList.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/StarTrek.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/Avengers_Endgame_poster.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/Forrest_Gump_poster.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/IT.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/Jaws.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/Parasite.jpeg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/RushHour3.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/theNotebook.jpg",
        "https://raw.githubusercontent.com/moolivieJHB/DataComponents/main/titanic.png",
    
    ]
    selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=400)

    if selectedImageUrl is not None:
        st.image(selectedImageUrl)

if __name__ == "__main__":
    carousel()
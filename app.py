import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import os
import random
import requests


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Cricket Face Analyzer",
    page_icon="🏏",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        text-align: center;
        margin-top: 20px;
    }

    .prediction {
        font-size: 32px;
        font-weight: 700;
    }

    .confidence {
        font-size: 22px;
        margin-top: 10px;
    }

    .bio-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
        line-height: 1.7;
    }

    .profile-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .wiki-button {
        font-size: 18px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("cricket_face_model.pkl")
    hog_settings = joblib.load("hog_settings.pkl")

    return model, hog_settings


model, hog_settings = load_model()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

@st.cache_resource
def load_face_detector():

    cascade_path = cv2.data.haarcascades + \
        "haarcascade_frontalface_default.xml"

    detector = cv2.CascadeClassifier(cascade_path)

    return detector


face_cascade = load_face_detector()


# ============================================================
# PLAYER NAME MAPPING
# ============================================================

player_names = {

    "Player_1": "Ambati Rayudu",
    "Player_2": "Hardik Pandya",
    "Player_3": "Jasprit Bumrah",
    "Player_4": "Kapil Dev",
    "Player_5": "M.S. Dhoni",
    "Player_6": "Ravindra Jadeja",
    "Player_7": "Rohit Sharma",
    "Player_8": "Ruturaj Gaikwad",
    "Player_9": "Sachin Tendulkar",
    "Player_10": "Virat Kohli"

}


# ============================================================
# WIKIPEDIA LINKS
# ============================================================

wikipedia_links = {

    "Ambati Rayudu":
        "https://en.wikipedia.org/wiki/Ambati_Rayudu",

    "Hardik Pandya":
        "https://en.wikipedia.org/wiki/Hardik_Pandya",

    "Jasprit Bumrah":
        "https://en.wikipedia.org/wiki/Jasprit_Bumrah",

    "Kapil Dev":
        "https://en.wikipedia.org/wiki/Kapil_Dev",

    "M.S. Dhoni":
        "https://en.wikipedia.org/wiki/MS_Dhoni",

    "Ravindra Jadeja":
        "https://en.wikipedia.org/wiki/Ravindra_Jadeja",

    "Rohit Sharma":
        "https://en.wikipedia.org/wiki/Rohit_Sharma",

    "Ruturaj Gaikwad":
        "https://en.wikipedia.org/wiki/Ruturaj_Gaikwad",

    "Sachin Tendulkar":
        "https://en.wikipedia.org/wiki/Sachin_Tendulkar",

    "Virat Kohli":
        "https://en.wikipedia.org/wiki/Virat_Kohli"

}


# ============================================================
# PLAYER BASIC INFORMATION
# ============================================================

player_info = {

    "Ambati Rayudu": {
        "role": "Batter",
        "batting": "Right-handed"
    },

    "Hardik Pandya": {
        "role": "All-rounder",
        "batting": "Right-handed"
    },

    "Jasprit Bumrah": {
        "role": "Fast Bowler",
        "batting": "Right-handed"
    },

    "Kapil Dev": {
        "role": "All-rounder",
        "batting": "Right-handed"
    },

    "M.S. Dhoni": {
        "role": "Wicketkeeper-Batter",
        "batting": "Right-handed"
    },

    "Ravindra Jadeja": {
        "role": "All-rounder",
        "batting": "Left-handed"
    },

    "Rohit Sharma": {
        "role": "Batter",
        "batting": "Right-handed"
    },

    "Ruturaj Gaikwad": {
        "role": "Batter",
        "batting": "Right-handed"
    },

    "Sachin Tendulkar": {
        "role": "Batter",
        "batting": "Right-handed"
    },

    "Virat Kohli": {
        "role": "Batter",
        "batting": "Right-handed"
    }

}


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = os.path.join(
    "dataset",
    "archive",
    "indian cricketer"
)


# ============================================================
# FIND PLAYER IMAGE
# ============================================================

def find_player_image(player_name):

    folder_name = player_name

    # Actual dataset folder name
    if player_name == "M.S. Dhoni":
        folder_name = "M.S. Dhoni1"

    player_folder = os.path.join(
        DATASET_PATH,
        folder_name
    )

    if not os.path.exists(player_folder):
        return None

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    image_files = []

    for file in os.listdir(player_folder):

        if file.lower().endswith(image_extensions):

            image_files.append(
                os.path.join(player_folder, file)
            )

    if len(image_files) == 0:
        return None

    return random.choice(image_files)


# ============================================================
# GET LONG WIKIPEDIA BIOGRAPHY
# ============================================================

@st.cache_data
def get_wikipedia_biography(player_name):

    if player_name not in wikipedia_links:
        return None

    url = wikipedia_links[player_name]

    try:

        api_url = "https://en.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "titles": player_name,
            "format": "json",
            "formatversion": "2",
            "redirects": "1"
        }

        headers = {
            "User-Agent":
                "CricketFaceAnalyzer/1.0 "
                "(educational project)"
        }

        response = requests.get(
            api_url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        pages = data["query"]["pages"]

        if len(pages) == 0:
            return None

        page = pages[0]

        biography = page.get(
            "extract",
            ""
        )

        if biography.strip() == "":
            return None

        return biography

    except Exception as e:

        return None


# ============================================================
# EXTRACT HOG FEATURES
# ============================================================

def extract_hog_features(face):

    face = cv2.resize(
        face,
        (100, 100)
    )

    face = face / 255.0

    features = hog(
        face,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title(
    "🏏 Cricket Face Analyzer"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Face Analysis",
        "🏏 Player Profiles"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🏏 Cricket Face Analyzer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Computer Vision Based Cricket Face Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader(
            "📷 Face Detection"
        )

        st.write(
            "Detects a face from an uploaded image "
            "using the Haar Cascade algorithm."
        )

    with col2:

        st.subheader(
            "🧠 Feature Extraction"
        )

        st.write(
            "Extracts facial features using "
            "Histogram of Oriented Gradients (HOG)."
        )

    with col3:

        st.subheader(
            "🤖 Classification"
        )

        st.write(
            "Uses the trained machine learning model "
            "to identify the cricket player."
        )

    st.divider()

    st.header(
        "How the application works"
    )

    st.write(
        "1. Upload an image."
    )

    st.write(
        "2. The application detects the largest face."
    )

    st.write(
        "3. The detected face is resized to 100 × 100 pixels."
    )

    st.write(
        "4. HOG extracts facial features."
    )

    st.write(
        "5. The trained machine learning model "
        "performs classification."
    )

    st.write(
        "6. The application identifies the player."
    )

    st.write(
        "7. The player's image and information are displayed."
    )

    st.write(
        "8. A long biography is retrieved from Wikipedia."
    )

    st.info(
        "Go to Face Analysis to test the model."
    )


# ============================================================
# FACE ANALYSIS PAGE
# ============================================================

elif page == "🔍 Face Analysis":

    st.markdown(
        '<div class="main-title">'
        '🔍 Face Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload an image to identify the cricket player'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp"
        ]
    )

    if uploaded_file is not None:

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        # ====================================================
        # CONVERT TO GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        # ====================================================
        # DETECT FACES
        # ====================================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:

            st.error(
                "❌ No face detected. "
                "Please upload a clearer image."
            )

        else:

            # ====================================================
            # FIND LARGEST FACE
            # ====================================================

            largest_face = max(
                faces,
                key=lambda rect:
                rect[2] * rect[3]
            )

            x, y, w, h = largest_face

            face = gray[
                y:y + h,
                x:x + w
            ]

            # ====================================================
            # EXTRACT HOG FEATURES
            # ====================================================

            features = extract_hog_features(
                face
            )

            features = features.reshape(
                1,
                -1
            )

            # ====================================================
            # MODEL PREDICTION
            # ====================================================

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = (
                np.max(probabilities) * 100
            )

            # ====================================================
            # CONVERT MODEL LABEL TO PLAYER NAME
            # ====================================================

            player_name = player_names.get(
                str(prediction),
                str(prediction)
            )

            # ====================================================
            # PLAYER INFORMATION
            # ====================================================

            info = player_info.get(
                player_name,
                {}
            )

            # ====================================================
            # DRAW FACE RECTANGLE
            # ====================================================

            result_image = image_array.copy()

            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # ====================================================
            # CROP DETECTED FACE
            # ====================================================

            detected_face = image_array[
                y:y + h,
                x:x + w
            ]

            # ====================================================
            # DISPLAY UPLOADED IMAGE
            # ====================================================

            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "📷 Uploaded Image"
                )

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "🎯 Detected Face"
                )

                st.image(
                    detected_face,
                    use_container_width=True
                )

            st.divider()

            # ====================================================
            # DETECTION RESULT
            # ====================================================

            st.subheader(
                "🔎 Detection Result"
            )

            st.image(
                result_image,
                caption="Detected face",
                use_container_width=True
            )

            # ====================================================
            # PLAYER NAME + CONFIDENCE
            # ====================================================

            st.markdown(
                f"""
                <div class="result-box">

                    <div class="prediction">
                        🏏 {player_name}
                    </div>

                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                int(min(confidence, 100))
            )

            # ====================================================
            # CONFIDENCE MESSAGE
            # ====================================================

            if confidence >= 70:

                st.success(
                    "✅ High confidence prediction."
                )

            elif confidence >= 50:

                st.warning(
                    "⚠️ Moderate confidence prediction."
                )

            else:

                st.warning(
                    "⚠️ Low confidence prediction. "
                    "Try a clearer image with the face "
                    "facing the camera."
                )

            # ====================================================
            # PLAYER INFORMATION
            # ====================================================

            st.divider()

            st.header(
                f"🏏 About {player_name}"
            )

            # ====================================================
            # PLAYER IMAGE
            # ====================================================

            player_image = find_player_image(
                player_name
            )

            col1, col2 = st.columns(
                [1, 2]
            )

            with col1:

                if player_image is not None:

                    st.image(
                        player_image,
                        caption=player_name,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "Player image not found "
                        "in the dataset."
                    )

            with col2:

                st.subheader(
                    "👤 Player Details"
                )

                if info:

                    st.write(
                        f"**Role:** {info.get('role', 'N/A')}"
                    )

                    st.write(
                        f"**Batting:** "
                        f"{info.get('batting', 'N/A')}"
                    )

                st.write(
                    f"**Recognized as:** "
                    f"{player_name}"
                )

            # ====================================================
            # WIKIPEDIA BIOGRAPHY
            # ====================================================

            st.divider()

            st.subheader(
                "📖 Biography"
            )

            with st.spinner(
                f"Loading {player_name}'s biography from Wikipedia..."
            ):

                biography = get_wikipedia_biography(
                    player_name
                )

            if biography:

                # ------------------------------------------------
                # Split biography into paragraphs
                # ------------------------------------------------

                paragraphs = biography.split("\n")

                for paragraph in paragraphs:

                    paragraph = paragraph.strip()

                    if paragraph:

                        st.markdown(
                            paragraph
                        )

            else:

                st.warning(
                    "Wikipedia biography could not be "
                    "loaded right now."
                )

                st.write(
                    "You can read the complete biography "
                    "using the Wikipedia link below."
                )

            # ====================================================
            # WIKIPEDIA LINK
            # ====================================================

            st.divider()

            st.subheader(
                "📚 Source"
            )

            wiki_url = wikipedia_links.get(
                player_name
            )

            if wiki_url:

                st.markdown(
                    f"""
                    <div class="wiki-button">

                    📖 [Read the full {player_name}
                    biography on Wikipedia]({wiki_url})

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ====================================================
            # MODEL INFORMATION
            # ====================================================

            st.divider()

            st.subheader(
                "🧠 Model Information"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Feature Method",
                    "HOG"
                )

            with col2:

                st.metric(
                    "Feature Count",
                    str(features.shape[1])
                )

            with col3:

                st.metric(
                    "Classifier",
                    type(model).__name__
                )


# ============================================================
# PLAYER PROFILES PAGE
# ============================================================

elif page == "🏏 Player Profiles":

    st.markdown(
        '<div class="main-title">'
        '🏏 Player Profiles'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore cricket player information'
        '</div>',
        unsafe_allow_html=True
    )

    selected_player = st.selectbox(
        "Select a player",
        list(player_names.values())
    )

    st.divider()

    # ========================================================
    # PLAYER IMAGE
    # ========================================================

    player_image = find_player_image(
        selected_player
    )

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        if player_image is not None:

            st.image(
                player_image,
                caption=selected_player,
                use_container_width=True
            )

        else:

            st.info(
                "Player image not found."
            )

    with col2:

        st.header(
            f"🏏 {selected_player}"
        )

        info = player_info.get(
            selected_player,
            {}
        )

        st.write(
            f"**Role:** "
            f"{info.get('role', 'N/A')}"
        )

        st.write(
            f"**Batting:** "
            f"{info.get('batting', 'N/A')}"
        )

    # ========================================================
    # WIKIPEDIA BIOGRAPHY
    # ========================================================

    st.divider()

    st.subheader(
        "📖 Biography"
    )

    with st.spinner(
        "Loading biography from Wikipedia..."
    ):

        biography = get_wikipedia_biography(
            selected_player
        )

    if biography:

        paragraphs = biography.split("\n")

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if paragraph:

                st.markdown(
                    paragraph
                )

    else:

        st.warning(
            "Wikipedia biography could not be loaded."
        )

    # ========================================================
    # WIKIPEDIA LINK
    # ========================================================

    st.divider()

    wiki_url = wikipedia_links.get(
        selected_player
    )

    if wiki_url:

        st.markdown(
            f"📖 [Read the full biography of "
            f"{selected_player} on Wikipedia]({wiki_url})"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Computer Vision Project | "
    "Haar Cascade + HOG + Machine Learning | "
    "Player information sourced from Wikipedia"
)

import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image
import os
import random


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
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 22px;
        margin-top: 10px;
    }

    .profile-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #dddddd;
        margin-top: 20px;
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

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

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
# PLAYER PROFILE DATA
# ============================================================

player_profiles = {

    "Ambati Rayudu": {
        "role": "Batter",
        "batting": "Right-handed",
        "about": (
            "Ambati Rayudu is an Indian cricketer known for his batting "
            "and his performances in domestic and international cricket."
        ),
        "career": (
            "Rayudu represented India in limited-overs cricket and played "
            "for several teams in domestic cricket and the Indian Premier League."
        ),
        "achievements": [
            "Represented India in international cricket",
            "Played for multiple IPL teams",
            "Known for his middle-order batting"
        ]
    },

    "Hardik Pandya": {
        "role": "All-rounder",
        "batting": "Right-handed",
        "about": (
            "Hardik Pandya is an Indian international cricketer known for "
            "his explosive batting and fast-medium bowling."
        ),
        "career": (
            "He has represented India in all three formats and has been "
            "an important all-rounder in limited-overs cricket."
        ),
        "achievements": [
            "Represented India in international cricket",
            "Won the 2024 T20 World Cup with India",
            "Successful IPL all-rounder"
        ]
    },

    "Jasprit Bumrah": {
        "role": "Fast Bowler",
        "batting": "Right-handed",
        "about": (
            "Jasprit Bumrah is an Indian international fast bowler known "
            "for his accuracy, pace and unusual bowling action."
        ),
        "career": (
            "He has become one of India's leading fast bowlers across "
            "Test, ODI and T20I cricket."
        ),
        "achievements": [
            "Represented India in all three formats",
            "Won the 2024 T20 World Cup with India",
            "Known for exceptional death bowling"
        ]
    },

    "Kapil Dev": {
        "role": "All-rounder",
        "batting": "Right-handed",
        "about": (
            "Kapil Dev is a legendary Indian cricketer and one of India's "
            "greatest all-rounders."
        ),
        "career": (
            "He captained India to its first Cricket World Cup victory "
            "in 1983."
        ),
        "achievements": [
            "Captain of India's 1983 World Cup winning team",
            "One of India's greatest all-rounders",
            "Former Indian Test and ODI captain"
        ]
    },

    "M.S. Dhoni": {
        "role": "Wicketkeeper-Batter",
        "batting": "Right-handed",
        "about": (
            "Mahendra Singh Dhoni is a former Indian international "
            "cricketer and one of the most successful captains in cricket."
        ),
        "career": (
            "Dhoni captained India to major international tournament "
            "victories and had a highly successful IPL career."
        ),
        "achievements": [
            "Captain of the 2007 T20 World Cup winning team",
            "Captain of the 2011 Cricket World Cup winning team",
            "Captain of the 2013 Champions Trophy winning team",
            "Multiple IPL titles with Chennai Super Kings"
        ]
    },

    "Ravindra Jadeja": {
        "role": "All-rounder",
        "batting": "Left-handed",
        "about": (
            "Ravindra Jadeja is an Indian international cricketer known "
            "for his left-arm spin bowling, batting and fielding."
        ),
        "career": (
            "He has been an important all-rounder for India in Test and "
            "limited-overs cricket."
        ),
        "achievements": [
            "Represented India in all three formats",
            "Won the 2024 T20 World Cup with India",
            "Known for outstanding fielding"
        ]
    },

    "Rohit Sharma": {
        "role": "Batter",
        "batting": "Right-handed",
        "about": (
            "Rohit Sharma is an Indian international cricketer known "
            "for his elegant batting and leadership."
        ),
        "career": (
            "He has represented India across formats and has captained "
            "India in major international tournaments."
        ),
        "achievements": [
            "Captain of India's 2024 T20 World Cup winning team",
            "Multiple-time IPL champion",
            "Holds the record for the highest individual ODI score"
        ]
    },

    "Ruturaj Gaikwad": {
        "role": "Batter",
        "batting": "Right-handed",
        "about": (
            "Ruturaj Gaikwad is an Indian cricketer known for his "
            "technically sound batting."
        ),
        "career": (
            "He has represented India in limited-overs cricket and has "
            "been a successful batter in the Indian Premier League."
        ),
        "achievements": [
            "Represented India in international cricket",
            "Successful IPL batter",
            "Known for consistent top-order batting"
        ]
    },

    "Sachin Tendulkar": {
        "role": "Batter",
        "batting": "Right-handed",
        "about": (
            "Sachin Tendulkar is a legendary Indian cricketer widely "
            "regarded as one of the greatest batters in cricket history."
        ),
        "career": (
            "He represented India for more than two decades and achieved "
            "numerous batting records during his international career."
        ),
        "achievements": [
            "Won the 2011 Cricket World Cup with India",
            "First male cricketer to score a double century in ODI cricket",
            "100 international centuries",
            "Bharat Ratna recipient"
        ]
    },

    "Virat Kohli": {
        "role": "Batter",
        "batting": "Right-handed",
        "about": (
            "Virat Kohli is an Indian international cricketer known for "
            "his batting, fitness and leadership."
        ),
        "career": (
            "He has represented India across formats and has been one of "
            "the leading run-scorers in modern international cricket."
        ),
        "achievements": [
            "Won the 2011 Cricket World Cup with India",
            "Won the 2024 T20 World Cup with India",
            "Multiple ICC awards",
            "Former captain of the Indian cricket team"
        ]
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

    # Dataset uses M.S. Dhoni1
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
# SIDEBAR
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
        '<div class="main-title">🏏 Cricket Face Analyzer</div>',
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

        st.subheader("📷 Face Detection")

        st.write(
            "Detects a face from an uploaded image using "
            "the Haar Cascade algorithm."
        )

    with col2:

        st.subheader("🧠 Feature Extraction")

        st.write(
            "Extracts facial features using "
            "Histogram of Oriented Gradients (HOG)."
        )

    with col3:

        st.subheader("🤖 Classification")

        st.write(
            "Uses the trained machine learning model "
            "to identify the cricket player."
        )

    st.divider()

    st.header("How the application works")

    st.write("1. Upload an image.")

    st.write("2. The application detects the largest face.")

    st.write("3. The detected face is resized to 100 × 100 pixels.")

    st.write("4. HOG extracts facial features.")

    st.write("5. The trained model performs classification.")

    st.write("6. The application identifies the player.")

    st.write("7. The player's information is displayed.")

    st.info(
        "Use the Face Analysis page to test the trained model."
    )


# ============================================================
# FACE ANALYSIS PAGE
# ============================================================

elif page == "🔍 Face Analysis":

    st.markdown(
        '<div class="main-title">🔍 Face Analysis</div>',
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

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:

            st.error(
                "❌ No face detected. "
                "Please upload a clearer face image."
            )

        else:

            # Find largest face
            largest_face = max(
                faces,
                key=lambda rect: rect[2] * rect[3]
            )

            x, y, w, h = largest_face

            face = gray[
                y:y + h,
                x:x + w
            ]

            # ====================================================
            # HOG FEATURES
            # ====================================================

            features = extract_hog_features(face)

            features = features.reshape(
                1,
                -1
            )

            # ====================================================
            # PREDICTION
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

            profile = player_profiles.get(
                player_name
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
            # DETECTED FACE
            # ====================================================

            detected_face = image_array[
                y:y + h,
                x:x + w
            ]

            # ====================================================
            # DISPLAY IMAGES
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

            st.subheader(
                "🔎 Detection Result"
            )

            st.image(
                result_image,
                caption="Detected face",
                use_container_width=True
            )

            # ====================================================
            # PLAYER RESULT
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
                    "Try a clearer image."
                )

            # ====================================================
            # PLAYER INFORMATION
            # ====================================================

            if profile is not None:

                st.divider()

                st.header(
                    f"🏏 About {player_name}"
                )

                # ------------------------------------------------
                # PLAYER IMAGE
                # ------------------------------------------------

                player_image = find_player_image(
                    player_name
                )

                if player_image is not None:

                    col1, col2 = st.columns(
                        [1, 2]
                    )

                    with col1:

                        st.image(
                            player_image,
                            caption=player_name,
                            use_container_width=True
                        )

                    with col2:

                        st.subheader(
                            "👤 Player Information"
                        )

                        st.write(
                            profile["about"]
                        )

                        st.write(
                            f"**Role:** {profile['role']}"
                        )

                        st.write(
                            f"**Batting:** {profile['batting']}"
                        )

                else:

                    st.subheader(
                        "👤 Player Information"
                    )

                    st.write(
                        profile["about"]
                    )

                    st.write(
                        f"**Role:** {profile['role']}"
                    )

                    st.write(
                        f"**Batting:** {profile['batting']}"
                    )

                # ------------------------------------------------
                # CAREER
                # ------------------------------------------------

                st.subheader(
                    "📋 Career"
                )

                st.write(
                    profile["career"]
                )

                # ------------------------------------------------
                # ACHIEVEMENTS
                # ------------------------------------------------

                st.subheader(
                    "🏆 Achievements"
                )

                for achievement in profile["achievements"]:

                    st.write(
                        f"• {achievement}"
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
        '<div class="main-title">🏏 Player Profiles</div>',
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
        list(player_profiles.keys())
    )

    profile = player_profiles[
        selected_player
    ]

    st.divider()

    # Player image
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

    with col2:

        st.header(
            f"🏏 {selected_player}"
        )

        st.subheader(
            "Role"
        )

        st.write(
            profile["role"]
        )

        st.subheader(
            "Batting"
        )

        st.write(
            profile["batting"]
        )

        st.subheader(
            "Biography"
        )

        st.write(
            profile["about"]
        )

    st.divider()

    st.subheader(
        "📋 Career"
    )

    st.write(
        profile["career"]
    )

    st.subheader(
        "🏆 Achievements"
    )

    for achievement in profile["achievements"]:

        st.write(
            f"• {achievement}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Computer Vision Project | "
    "Haar Cascade + HOG + Machine Learning"
)

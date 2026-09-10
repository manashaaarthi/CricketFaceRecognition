import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image


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
# PLAYER PROFILE DATA
# ============================================================
# Anonymous labels are used for the ML prediction.
# Profiles are accessed separately through the dropdown.
# ============================================================

player_profiles = {

    "Player_1": {
        "title": "Player Profile 1",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_2": {
        "title": "Player Profile 2",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_3": {
        "title": "Player Profile 3",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_4": {
        "title": "Player Profile 4",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_5": {
        "title": "Player Profile 5",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_6": {
        "title": "Player Profile 6",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_7": {
        "title": "Player Profile 7",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_8": {
        "title": "Player Profile 8",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_9": {
        "title": "Player Profile 9",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    },

    "Player_10": {
        "title": "Player Profile 10",
        "role": "Indian Cricketer",
        "bio": "This profile contains biographical and career information for the selected player.",
        "career": "Professional cricket career with experience in competitive matches.",
        "achievements": [
            "Represented teams in professional cricket",
            "Participated in major cricket competitions",
            "Contributed to team performances"
        ]
    }
}


# ============================================================
# FUNCTION: EXTRACT HOG FEATURES
# ============================================================

def extract_hog_features(face):

    face = cv2.resize(face, (100, 100))

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

st.sidebar.title("🏏 Cricket Face Analyzer")

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
        '<div class="subtitle">Computer Vision Based Cricket Face Classification</div>',
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
            "Uses Logistic Regression to classify "
            "the detected face into one of the trained classes."
        )

    st.divider()

    st.header("How the application works")

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
        "4. HOG extracts 4356 facial features."
    )

    st.write(
        "5. Logistic Regression performs classification."
    )

    st.write(
        "6. The application displays the predicted class and confidence."
    )

    st.info(
        "Use the Face Analysis page to test the trained computer vision model."
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
        '<div class="subtitle">Upload an image to analyze the detected face</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

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
                "❌ No face detected. Please upload a clearer face image."
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

            # Extract HOG
            features = extract_hog_features(face)

            features = features.reshape(1, -1)

            # Prediction
            prediction = model.predict(features)[0]

            # Confidence
            probabilities = model.predict_proba(features)[0]

            confidence = np.max(probabilities) * 100

            # Draw rectangle
            result_image = image_array.copy()

            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            # Crop face for display
            detected_face = image_array[
                y:y + h,
                x:x + w
            ]

            # =================================================
            # DISPLAY IMAGES
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("📷 Uploaded Image")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.subheader("🎯 Detected Face")

                st.image(
                    detected_face,
                    use_container_width=True
                )

            st.divider()

            st.subheader("🔎 Detection Result")

            st.image(
                result_image,
                caption="Detected face",
                use_container_width=True
            )

            # =================================================
            # PREDICTION
            # =================================================

            st.markdown(
                f"""
                <div class="result-box">

                    <div class="prediction">
                        Prediction: {prediction}
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

            if confidence >= 70:

                st.success(
                    "High confidence prediction."
                )

            elif confidence >= 50:

                st.warning(
                    "Moderate confidence prediction."
                )

            else:

                st.warning(
                    "Low confidence prediction. "
                    "Try a clearer image with the face facing the camera."
                )

            st.divider()

            st.subheader("🧠 Model Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Feature Method",
                    "HOG"
                )

            with col2:
                st.metric(
                    "Feature Count",
                    "4356"
                )

            with col3:
                st.metric(
                    "Classifier",
                    "Logistic Regression"
                )


# ============================================================
# PLAYER PROFILE PAGE
# ============================================================

elif page == "🏏 Player Profiles":

    st.markdown(
        '<div class="main-title">🏏 Player Profiles</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore cricket player information</div>',
        unsafe_allow_html=True
    )

    selected_player = st.selectbox(
        "Select a player profile",
        list(player_profiles.keys())
    )

    profile = player_profiles[selected_player]

    st.divider()

    st.markdown(
        '<div class="profile-box">',
        unsafe_allow_html=True
    )

    st.header(
        f"🏏 {profile['title']}"
    )

    st.subheader("Role")

    st.write(
        profile["role"]
    )

    st.subheader("Biography")

    st.write(
        profile["bio"]
    )

    st.subheader("Career")

    st.write(
        profile["career"]
    )

    st.subheader("🏆 Achievements")

    for achievement in profile["achievements"]:

        st.write(
            f"• {achievement}"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Computer Vision Project | Haar Cascade + HOG + Logistic Regression"
)

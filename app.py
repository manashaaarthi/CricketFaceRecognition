import streamlit as st
import cv2
import numpy as np
import joblib

from PIL import Image
from skimage.feature import hog


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Indian Cricket Face Classification",
    page_icon="🏏",
    layout="centered"
)


# ==========================================================
# PLAYER MAPPING
# ==========================================================

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


# ==========================================================
# TITLE
# ==========================================================

st.title("🏏 Indian Cricket Face Classification")

st.write(
    "Upload an image containing a face to identify "
    "the corresponding trained class."
)


# ==========================================================
# LOAD MODEL
# ==========================================================

try:

    model = joblib.load(
        "cricket_face_model.pkl"
    )

    hog_settings = joblib.load(
        "hog_settings.pkl"
    )

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.error(str(e))

    st.stop()


# ==========================================================
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():

    st.error(
        "Face detection model could not be loaded."
    )

    st.stop()


# ==========================================================
# IMAGE UPLOAD
# ==========================================================

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


# ==========================================================
# PROCESS IMAGE
# ==========================================================

if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    image_array = np.array(
        image
    )


    # ======================================================
    # DISPLAY ORIGINAL IMAGE
    # ======================================================

    st.subheader("Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )


    # ======================================================
    # CONVERT TO OPENCV
    # ======================================================

    cv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    gray = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY
    )


    # ======================================================
    # FACE DETECTION
    # ======================================================

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )


    # ======================================================
    # NO FACE DETECTED
    # ======================================================

    if len(faces) == 0:

        st.warning(
            "⚠️ No face detected in the uploaded image."
        )

        st.info(
            "Please upload a clear image with a visible face."
        )


    else:

        st.success(
            f"✅ Face detected: {len(faces)}"
        )


        # ==================================================
        # SELECT LARGEST FACE
        # ==================================================

        x, y, w, h = max(
            faces,
            key=lambda rect: rect[2] * rect[3]
        )


        # ==================================================
        # CROP FACE
        # ==================================================

        face = gray[
            y:y + h,
            x:x + w
        ]


        # ==================================================
        # RESIZE FACE
        # ==================================================

        image_size = tuple(
            hog_settings["image_size"]
        )

        face = cv2.resize(
            face,
            image_size
        )


        # ==================================================
        # NORMALIZE
        # ==================================================

        face = face / 255.0


        # ==================================================
        # HOG FEATURE EXTRACTION
        # ==================================================

        hog_features = hog(
            face,
            orientations=hog_settings[
                "orientations"
            ],
            pixels_per_cell=hog_settings[
                "pixels_per_cell"
            ],
            cells_per_block=hog_settings[
                "cells_per_block"
            ],
            block_norm=hog_settings[
                "block_norm"
            ]
        )


        # ==================================================
        # RESHAPE FEATURES
        # ==================================================

        features = hog_features.reshape(
            1,
            -1
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        prediction = model.predict(
            features
        )

        predicted_class = prediction[0]


        # ==================================================
        # CONVERT CLASS TO PLAYER NAME
        # ==================================================

        predicted_player = player_names.get(
            predicted_class,
            predicted_class
        )


        # ==================================================
        # CONFIDENCE
        # ==================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                features
            )

            confidence = (
                np.max(probabilities) * 100
            )

        else:

            confidence = 0


        # ==================================================
        # DISPLAY DETECTED FACE
        # ==================================================

        face_display = (
            face * 255
        ).astype(
            np.uint8
        )

        st.subheader(
            "Detected Face"
        )

        st.image(
            face_display,
            caption="Face used for classification",
            width=250
        )


        # ==================================================
        # DISPLAY PREDICTION
        # ==================================================

        st.subheader(
            "🏆 Prediction"
        )

        st.success(
            f"🏏 Player: {predicted_player}"
        )


        # ==================================================
        # DISPLAY CONFIDENCE
        # ==================================================

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # ==================================================
        # CONFIDENCE BAR
        # ==================================================

        st.progress(
            min(
                int(confidence),
                100
            )
        )


        # ==================================================
        # DRAW FACE RECTANGLE
        # ==================================================

        result_image = cv_image.copy()

        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )


        # ==================================================
        # DISPLAY DETECTION RESULT
        # ==================================================

        result_image = cv2.cvtColor(
            result_image,
            cv2.COLOR_BGR2RGB
        )

        st.subheader(
            "Face Detection Result"
        )

        st.image(
            result_image,
            caption="Detected face",
            use_container_width=True
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Computer Vision Project | "
    "Haar Cascade + HOG + Logistic Regression"
)
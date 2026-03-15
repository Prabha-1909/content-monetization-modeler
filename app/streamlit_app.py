import streamlit as st
import joblib
import numpy as np

st.markdown("""
<style>
.stApp {
    background-color: #FFE5E5;
}
</style>
""", unsafe_allow_html=True)

st.title("YouTube Ad Revenue Prediction App")
st.image("https://storage.googleapis.com/website-production/uploads/2020/02/youtube-revenue-data-1540x768.png", width=300)
st.subheader("Model Performance")

st.write("R² Score: 0.95")
st.write("MAE: 3.08")
st.write("RMSE: 13.47")
st.write("Predict estimated ad revenue based on video performance metrics.")
st.subheader("Input Features Explanation")

st.write("""
- **Views**: Number of times the video was watched
- **Likes**: Number of likes received
- **Comments**: Number of user comments
- **Watch Time**: Total minutes watched
- **Video Length**: Duration of the video
- **Subscribers**: Channel subscriber count
""")

# Load trained model
model = joblib.load("models/revenue_model.pkl")

st.title("YouTube Ad Revenue Predictor")

st.write("Enter video details to estimate ad revenue")

views = st.number_input("Views", min_value=1)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
watch_time = st.number_input("Watch Time (minutes)", min_value=0)
video_length = st.number_input("Video Length (minutes)", min_value=0)
subscribers = st.number_input("Subscribers", min_value=0)

# Derived features
engagement_rate = (likes + comments) / views
watch_time_per_view = watch_time / views

if st.button("Predict Revenue"):

    features = np.array([[views,
                          likes,
                          comments,
                          watch_time,
                          video_length,
                          subscribers,
                          0, 0, 0,
                          engagement_rate,
                          watch_time_per_view]])

    prediction = model.predict(features)

    st.success(f"Estimated Ad Revenue: ${prediction[0]:.2f}")
import streamlit as st
import joblib

# Load trained model
model = joblib.load("model/spam_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧"
)

# Application heading
st.title("📧 Email Spam Detector")

st.write(
    "Enter an email below to check "
    "whether it is Spam or Ham."
)

# Input fields
subject = st.text_input("Email Subject")

body = st.text_area("Email Content")

# Analyze button
if st.button("Analyze Email"):

    if not subject.strip() and not body.strip():
        st.warning("Please enter an email.")

    else:
        # Combine subject and body
        email = subject + " " + body

        # Predict
        prediction = model.predict([email])[0]

        # Calculate probability
        probabilities = model.predict_proba([email])[0]

        spam_index = list(model.classes_).index("spam")

        spam_probability = probabilities[spam_index] * 100

        # Display result
        if prediction == "spam":
            st.error("🚨 This email is SPAM!")

        else:
            st.success("✅ This email is HAM!")

        st.write(
            f"Spam Probability: {spam_probability:.2f}%"
        )
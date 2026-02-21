import streamlit as st
import joblib
import pandas as pd

# Page setup
st.set_page_config(page_title="AI Complaint Classifier", page_icon="🤖")

# Load model and vectorizer
@st.cache_resource
def load_ai():
    model = joblib.load('complaint_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_ai()

# Initialize session history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("📂 Customer Complaint Categorizer")
st.markdown("Enter a financial complaint to classify it into professional categories.")

user_input = st.text_area("Complaint Text:", placeholder="e.g., I have an unauthorized charge on my credit card...")

if st.button("Classify Complaint"):
    if user_input:
        # Prediction
        vec = tfidf.transform([user_input])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)
        confidence = probs.max() * 100

        # UI Results
        st.divider()
        st.subheader(f"Result: {prediction.replace('_', ' ').title()}")
        
        if confidence < 55: # Slightly higher threshold for research quality
            st.warning(f"Confidence: {confidence:.2f}% (Human Review Recommended)")
        else:
            st.success(f"Confidence: {confidence:.2f}% (High Certainty)")
        
        # Save to history
        st.session_state.history.append({
            "Complaint": user_input[:50] + "...", 
            "Category": prediction, 
            "Confidence": f"{confidence:.2f}%"
        })
    else:
        st.error("Please enter a complaint first.")

# Display History and Download Button
if st.session_state.history:
    st.divider()
    st.subheader("Session History")
    df_history = pd.DataFrame(st.session_state.history)
    st.table(df_history)
    
    csv = df_history.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Research Report", data=csv, file_name='complaint_report.csv', mime='text/csv')
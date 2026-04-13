import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Page setup
st.set_page_config(
    page_title="AI Complaint Classifier",
    page_icon="🤖",
    layout="wide"
)

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

# ── Header ──────────────────────────────────────────────────────────────
st.title("📂 Financial Complaint Triage System")
st.markdown("**HITL-enabled NLP classifier** | 85%+ accuracy | 5-class CFPB complaint routing")
st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Classify Complaint", "📋 Session History", "📊 Model Performance"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1 — Classify
# ════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Enter Complaint")
        user_input = st.text_area(
            "Complaint Text:",
            height=180,
            placeholder="e.g., I have an unauthorized charge on my credit card that I did not make..."
        )

        classify_btn = st.button("🚀 Classify Complaint", width="stretch")

    with col2:
        st.subheader("Classification Result")

        if classify_btn:
            if user_input.strip():
                # Prediction
                vec = tfidf.transform([user_input])
                prediction = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]
                confidence = probs.max() * 100

                # ── HITL Escalation Badge ──
                if confidence >= 75:
                    st.success("✅ AUTO-APPROVED")
                    hitl_status = "Auto-Approved"
                elif confidence >= 55:
                    st.warning("⚠️ HUMAN REVIEW RECOMMENDED")
                    hitl_status = "Human Review"
                else:
                    st.error("🚨 CRITICAL ESCALATION")
                    hitl_status = "Critical Escalation"

                # ── Primary Result ──
                st.metric(
                    label="Predicted Category",
                    value=prediction.replace('_', ' ').title(),
                    delta=f"{confidence:.1f}% confidence"
                )

                st.divider()

                # ── Top 3 Predictions with progress bars ──
                st.markdown("**Top 3 Predictions:**")
                top3_idx = probs.argsort()[-3:][::-1]
                for idx in top3_idx:
                    label = model.classes_[idx].replace('_', ' ').title()
                    score = probs[idx] * 100
                    st.progress(
                        int(score),
                        text=f"{label}: {score:.1f}%"
                    )

                # Save to history
                st.session_state.history.append({
                    "Complaint": user_input[:60] + "..." if len(user_input) > 60 else user_input,
                    "Category": prediction.replace('_', ' ').title(),
                    "Confidence": f"{confidence:.1f}%",
                    "HITL Status": hitl_status
                })

            else:
                st.error("Please enter a complaint first.")

        else:
            st.info("Enter a complaint on the left and click Classify.")

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Session History
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Session History")

    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)

        # Summary metrics
        total = len(df_history)
        auto = len(df_history[df_history['HITL Status'] == 'Auto-Approved'])
        review = len(df_history[df_history['HITL Status'] == 'Human Review'])
        critical = len(df_history[df_history['HITL Status'] == 'Critical Escalation'])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Classified", total)
        m2.metric("Auto-Approved", auto)
        m3.metric("Human Review", review)
        m4.metric("Critical Escalation", critical)

        st.divider()
        st.dataframe(df_history, width="stretch")

        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Report as CSV",
            data=csv,
            file_name='complaint_triage_report.csv',
            mime='text/csv',
            width="stretch"
        )

        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No complaints classified yet. Go to the Classify tab to get started.")

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance")

    # Summary stats
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Model", "Logistic Regression")
    s2.metric("Accuracy", "85.42%")
    s3.metric("Training Samples", "25,000")
    s4.metric("HITL Threshold", "55%")

    st.divider()

    # Confusion matrix
    st.markdown("**Confusion Matrix**")
    try:
        img = mpimg.imread('research_confusion_matrix.png')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(img)
        ax.axis('off')
        st.pyplot(fig)
    except FileNotFoundError:
        st.warning("Confusion matrix not found. Run classify.py first to generate it.")

    st.divider()

    # Per-class performance table
    st.markdown("**Per-Class Performance (Test Set)**")
    perf_data = {
        "Category": ["Credit Card", "Credit Reporting", "Debt Collection", "Mortgages & Loans", "Retail Banking"],
        "Precision": ["0.84", "0.84", "0.85", "0.87", "0.87"],
        "Recall":    ["0.83", "0.82", "0.80", "0.89", "0.92"],
        "F1-Score":  ["0.84", "0.83", "0.82", "0.88", "0.90"],
        "Support":   ["1000", "1000", "1000", "1000", "1000"]
    }
    st.dataframe(pd.DataFrame(perf_data), width="stretch", hide_index=True)

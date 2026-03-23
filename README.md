# 📂 Financial Complaint Triage AI
> **An Intelligent NLP Pipeline for Automated Dispute Routing & HITL Oversight**

**🔗 [Live Demo on Streamlit](https://github.com/Ekta-13/Financial-Complaint-Triage-AI)**

## 🎯 What it is
I engineered an end-to-end **Natural Language Processing (NLP)** system to automate the triage of 40,000+ consumer complaints from the CFPB database. The system converts messy, unstructured text into structured categories (e.g., Mortgages, Debt Collection, Credit Cards), allowing financial institutions to prioritize urgent disputes instantly.

## ⚙️ How it works
The system follows a professional **Data Engineering Pipeline**:
1. **Preprocessing:** Uses **NLTK** for deep text cleaning—including lemmatization and bi-gram extraction to capture context (e.g., "not received" vs. "received").
2. **Feature Engineering:** Converts text into numerical signals using **TF-IDF Vectorization**, prioritizing unique keywords that define specific financial products.
3. **The Classifier:** A **Logistic Regression** model optimized for high-speed, high-accuracy multi-class classification.
4. **Human-in-the-Loop (HITL):** A custom confidence-threshold logic flags ambiguous complaints (confidence <55%) for manual review, ensuring 100% reliability for complex cases.

## 🚀 Key Features
* **⚡ High-Throughput Triage:** Capable of classifying thousands of documents per second.
* **🛡️ Reliability Guardrails:** Integrated HITL system to mitigate "AI hallucination" in high-stakes financial data.
* **📊 Visual Monitoring:** Interactive Streamlit dashboard featuring a live **Confusion Matrix** to monitor model performance.
* **📦 Portable Logic:** Model and vectorizer are serialized via **Joblib** for instant deployment without retraining.

## 🛠️ Tech Stack
* **Core NLP:** Python, NLTK, Scikit-Learn
* **Mathematics:** TF-IDF, NumPy
* **App Framework:** Streamlit
* **Data Management:** Pandas

## 🧠 Strategic Engineering (The "Anti-AI" Flex)
* **Handling Class Imbalance:** Used **Stratified Splitting** to ensure the model performs accurately even on minority categories like "Payday Loans."
* **Precision-First Design:** Optimized for **Precision** to ensure that complaints routed to specialized departments are correctly labeled, reducing operational "ping-pong" between teams.

## 📉 System Architecture
![NLP Triage Flowchart](<img width="452" height="478" alt="image" src="https://github.com/user-attachments/assets/bc65d503-3c11-4c74-9d92-c02347ea32d3" />
)

## ⚡ Quick Start
```bash
# Clone the repository
git clone [https://github.com/Ekta-13/Financial-Complaint-Triage-AI.git](https://github.com/Ekta-13/Financial-Complaint-Triage-AI.git)

# Install dependencies
pip install -r requirements.txt

# Run the Triage App
streamlit run app.py

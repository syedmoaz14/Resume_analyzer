import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# 📄 Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# 🧹 Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    custom_stopwords = set(stopwords.words('english')) - {'data', 'science', 'analysis', 'model', 'machine'}
    tokens = [word for word in tokens if word not in custom_stopwords]
    return ' '.join(tokens)

# 🎯 Streamlit App
st.title("📄 Resume Analyzer")
st.markdown("Upload your resume and paste a job description to check how well they match!")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_input = st.text_area("💼 Paste Job Description Here")

if uploaded_file and job_input:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_clean = preprocess(resume_text)
    job_clean = preprocess(job_input)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_clean, job_clean])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    st.success(f"✅ Match Score: {round(score * 100, 2)}%")
elif st.button("🔍 Analyze"):
    st.warning("Please upload a resume and paste a job description.")


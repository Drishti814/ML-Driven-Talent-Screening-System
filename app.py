import streamlit as st
import pickle
import spacy
from wordcloud import WordCloud
from docx import Document
import fitz  
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Attempt to load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Downloading the spaCy language model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Preprocess text
def preprocess_text(text):
    doc = nlp(text)
    clean = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(clean)

# Load model
with open('model_res.pkl', 'rb') as file:
    model_pipeline = pickle.load(file)

# Category mapping
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

# Predict category
def predict_category(text):
    cleaned_text = preprocess_text(text)
    prediction_id = model_pipeline.predict([cleaned_text])[0]
    return category_mapping.get(prediction_id, "Unknown")

# Extract text
def extract_text_from_pdf(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# Main app
def main():
    st.title("Advanced Resume Screening App", anchor=None)
    st.markdown('<p class="big-font">Welcome to the Advanced Resume Screening App!</p>', unsafe_allow_html=True)
    st.markdown("## Upload your resume or type in your details below:")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Your Resume", type=["txt", "pdf", "docx"])
    with col2:
        resume_text_manual = st.text_area("Type your resume here:")

    resume_text = ""
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            resume_text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(uploaded_file)

        # Job Description Section
        st.markdown("## Upload or Type Job Description:")
        jd_col1, jd_col2 = st.columns(2)

        with jd_col1:
            jd_file = st.file_uploader("Upload Job Description", type=["txt", "pdf", "docx"])
            jd_text = ""
            if jd_file is not None:
                if jd_file.type == "application/pdf":
                    jd_text = extract_text_from_pdf(jd_file)
                elif jd_file.type == "text/plain":
                    jd_text = jd_file.getvalue().decode("utf-8")
                elif jd_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    jd_text = extract_text_from_docx(jd_file)

        with jd_col2:
            jd_text_manual = st.text_area("Type Job Description here:")

        # Priority: manual > file
        if jd_text_manual.strip():
            jd_text = jd_text_manual

        def calculate_match_score(resume_text, jd_text):
            if not jd_text:
                return None
            vectorizer = TfidfVectorizer(stop_words="english")
            vectors = vectorizer.fit_transform([resume_text, jd_text])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return similarity * 100

    elif resume_text_manual:
        resume_text = resume_text_manual
        jd_text = ""  # No JD if typed resume

    if resume_text and st.button("Analyze Resume"):
        with st.spinner("Analyzing..."):
            category = predict_category(resume_text)
            st.markdown(f"## Predicted Category: **{category}**")

            if 'jd_text' in locals() and jd_text:
                match_score = calculate_match_score(resume_text, jd_text)
                st.markdown(f"### Resume–JD Match: **{match_score:.2f}%**")

            # WordCloud
            cleaned_text = preprocess_text(resume_text)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

if __name__ == "__main__":
    main()

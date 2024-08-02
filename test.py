import streamlit as st
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64


class MatchMap:
    def __init__(self):
        self.job_ad = None
        self.cvs = []

    def extract_text_from_file(self, file):
        if file.name.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file)
        elif file.name.lower().endswith('.docx'):
            return self.extract_text_from_docx(file)
        else:
            raise ValueError("Nid unterstützts Dateiformat")

    def extract_text_from_pdf(self, file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def extract_text_from_docx(self, file):
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def match_candidates(self):
        if not self.job_ad or not self.cvs:
            st.error("Bitte lade zerscht d Stelleazeig und d Läbensläuf ufe.")
            return

        vectorizer = TfidfVectorizer()
        job_ad_vector = vectorizer.fit_transform([self.job_ad])

        matches = []
        for cv_name, cv_text in self.cvs:
            cv_vector = vectorizer.transform([cv_text])
            similarity = cosine_similarity(job_ad_vector, cv_vector)[0][0]
            matches.append((cv_name, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches


def main():
    st.set_page_config(page_title="MatchMap", page_icon="🎯", layout="wide")
    st.title("MatchMap - KI-gstützti Läbenslauf-Vergliichig")

    app = MatchMap()

    st.sidebar.header("Dateie ufelade")
    job_ad_file = st.sidebar.file_uploader(
        "Stelleazeig ufelade", type=["pdf", "docx"])
    cv_files = st.sidebar.file_uploader(
        "Läbensläuf ufelade", type=["pdf", "docx"], accept_multiple_files=True)

    if job_ad_file:
        app.job_ad = app.extract_text_from_file(job_ad_file)
        st.sidebar.success("Stelleazeig erfolgriich ufeglade!")

    if cv_files:
        for cv_file in cv_files:
            cv_text = app.extract_text_from_file(cv_file)
            app.cvs.append((cv_file.name, cv_text))
        st.sidebar.success(f"{len(cv_files)} Läbensläuf erfolgriich ufeglade!")

    if st.button("Kandidate vergliiche"):
        if app.job_ad and app.cvs:
            matches = app.match_candidates()
            st.subheader("Priorisierti Lischte vo de Kandidate:")
            for i, (cv_name, similarity) in enumerate(matches, 1):
                st.write(f"{i}. {cv_name} - Überiistimmig: {similarity:.2f}")

                # Create a download button for each CV
                cv_text = next(cv[1] for cv in app.cvs if cv[0] == cv_name)
                b64 = base64.b64encode(cv_text.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="{cv_name}">Läbenslauf abelade</a>'
                st.markdown(href, unsafe_allow_html=True)

                st.write("---")

            # Add print button
            if st.button("Analyse drucke"):
                st.markdown(
                    """
                    <script>
                    function printAnalysis() {
                        window.print();
                    }
                    </script>
                    <button onclick="printAnalysis()">Drucke</button>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Bitte lade zerscht d Stelleazeig und d Läbensläuf ufe.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Über MatchMap")
    st.sidebar.info("""
    MatchMap isch e KI-gstützti Aawändig, wo automatisch Läbensläuf mit Stelleazeige vergliicht.
    Si generiert e priorisierti Lischte vo de passendste Kandidate.

    Verwendeti Technologie:
    - Python: Programmiersprach
    - Streamlit: Für d Erstellig vo de Webaawändig
    - PyPDF2: Für s Usläse vo Text us PDF-Dateie
    - python-docx: Für s Usläse vo Text us DOCX-Dateie
    - scikit-learn: Für d Textanalyse und de Vergliich (TfidfVectorizer und cosine_similarity)
    """)


if __name__ == "__main__":
    main()

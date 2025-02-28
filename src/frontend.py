import streamlit as st
import tempfile
import pandas as pd
from pipeline import NotesPipeline
import torch 

torch.classes.__path__ = [] # Goofy bugfix for torch

class Frontend:
    def __init__(self):
        st.title("Two Column Notes Generator")

    def create_local_copy(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name, mode='wb') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            return temp_file.name  # Returns the full path of the temp file

    def transform_into_dataframe(self, notes: dict):
        df = pd.DataFrame({
            "Concepts:": notes.keys(),
            "Key things to know:": notes.values()
        })
        return df

    def run(self):
        api_key_input_widget = st.sidebar.text_input("Cohere API KEY", type="password")
        pdf_uploader_widget = st.file_uploader("Upload your reading here (200MB):", type="pdf", accept_multiple_files=False)

        with st.form("app"):
            submitted = st.form_submit_button("Submit for generation")
            if submitted and pdf_uploader_widget and api_key_input_widget:
                path = self.create_local_copy(pdf_uploader_widget)
                with st.spinner("Generating notes", show_time=True):
                    pipeline = NotesPipeline(api_key_input_widget, path)
                    result = pipeline.invoke() 
                st.table(self.transform_into_dataframe(result))


# To run the frontend
if __name__ == "__main__":
    app = Frontend()
    app.run()

from pipeline import NotesPipeline
import streamlit as st 
import tempfile
import pandas as pd

class StreamlitUI: 
  def __init__(self): 
    pass 


def create_local_copy(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name, mode='wb') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name  # Returns the full path of the temp file

def transform_into_dataframe(notes: dict): 
  df = pd.DataFrame({
    "Concepts:" : notes.keys(), 
    "Key things to know: " : notes.values()
  })
  return df

st.title("Two Column Notes Generator")

api_key_input_widget = st.sidebar.text_input("Cohere API KEY", type="password")
pdf_uploader_widget = st.file_uploader("Upload your reading here (200MB):", type="pdf", accept_multiple_files=False)


with st.form("app"):
    submitted = st.form_submit_button("Submit for generation")
    if submitted and pdf_uploader_widget and api_key_input_widget:
      path = create_local_copy(pdf_uploader_widget)
      st.write("Creating notes for" + path)
      pipeline = NotesPipeline(api_key_input_widget, path)
      st.table(transform_into_dataframe(pipeline.invoke()))

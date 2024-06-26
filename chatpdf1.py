
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pdfreader import SimplePDFViewer, PageDoesNotExist
from langchain.document_loaders import DirectoryLoader


load_dotenv()
os.getenv("GOOGLE_API_KEY")
print(os.getenv("GOOGLE_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
vbb    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If context is insufficient or irrelevant, exercise your knowledge of Indian legal frameworks to provide actionable advice. 
    Avoid speculative content and prioritize realism and practicality.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "context" : docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])





def extract_text_from_pdfs(directory):
    raw_text = ''

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                viewer = SimplePDFViewer(file)

                try:
                    while True:
                        viewer.render()
                        content = viewer.canvas.strings
                        if content:
                            raw_text += ''.join(content)
                        viewer.next()
                except PageDoesNotExist:
                    pass

    return raw_text


def main():
    directory_path = 'Legaldata'  
    text = extract_text_from_pdfs(directory_path)
    # raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)
    get_vector_store(text_chunks)
    st.set_page_config("LawGenie2")
    st.header("LawGenie")
    st.info("Your Virtual Legal Expert")
    user_question = st.text_area("Ask your questions")

    if user_question:
        user_input(user_question)
       
if __name__ == "__main__":
    main()

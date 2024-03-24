import os
import streamlit as st
import openai
import pinecone
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
import pymongo

# Set up your API keys and credentials (replace with your actual values)
os.environ["OPENAI_API_KEY"] = "your api key"
GMAIL_USERNAME = "example@gmail.com"
GMAIL_PASSWORD = "mhzs fgaf zqdq aqvp"
pinecone.init(api_key='pinecone api key', environment='environment')

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["chatbot_db"]
collection = db["conversation_history"]

# Load the PDF documents, embeddings, vector store, language model, and chain
loader = PyPDFLoader('YourPdf.pdf')
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents=documents)
embeddings = OpenAIEmbeddings(model_name="text-embedding-ada-002")
index_name = "langchain-chatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
model_name = "gpt-4"
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# Function to get similar documents
def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Function to get the answer to a question
def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Function to translate text using OpenAI
def translate_text_with_openai(text, target_lang):
    translation_result = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Translate the following English text to {target_lang}: '{text}'",
        max_tokens=2000
    )
    return translation_result['choices'][0]['text'].strip()

# Function to send an email
def send_email(subject, body, recipient):
    try:
        # Set up the MIMEText and MIMEMultipart objects
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USERNAME
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Set up the SMTP server with a secure connection
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            # Login to the email account
            server.login(GMAIL_USERNAME, GMAIL_PASSWORD)
            # Send the email
            server.sendmail(GMAIL_USERNAME, recipient, msg.as_string())

        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")

# Streamlit app layout
st.title("Multilingual Question Answering with Email Option")

# Use st.form to create a form
with st.form("answer_form"):
    recipient_email = st.text_input("Your email address:")
    query = st.text_input("Enter your question:")

    target_languages = [
        "French",
        "German",
        "Spanish",
        "Hindi",
        "Kannada",
        "Urdu",
        "Tamil",
        "Telugu",
        "English",
        "Chinese",
        "Japanese",
        "Sanskrit"
    ]
    target_lang = st.selectbox("Choose target language:", target_languages)

    send_email_checkbox = st.checkbox("Send answer to my email")

    # "Get Answer" button to submit the form
    if st.form_submit_button("Get Answer"):
        answer = get_answer(query)
        translated_answer = translate_text_with_openai(answer, target_lang)
        st.write("Answer in English:", answer)
        st.write("Translated Answer:", translated_answer)

        if send_email_checkbox and recipient_email:
            # Email sending logic is only executed when the checkbox is selected and recipient email is provided
            email_subject = "Question Answer"
            email_body = f"Query: {query}\nAnswer: {answer}\nTranslated Answer: {translated_answer}"
            send_email(email_subject, email_body, recipient_email)

            # Store the query and answer in MongoDB
            document = {
                "query": query,
                "answer": answer,
                "translated_answer": translated_answer
            }
            collection.insert_one(document)
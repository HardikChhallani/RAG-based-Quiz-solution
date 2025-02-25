from dotenv import load_dotenv #Library for loading the Gemini api key 
import streamlit as st
import os
import langchain
import google.generativeai as genai
import pathlib
import textwrap
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
openai_api_key =os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatOpenAI(api_key=openai_api_key)

def get_response(question):
    prompt = ChatPromptTemplate.from_template("""You are expert in the field of traditional knowlegde. 
    The question with four inputs named as a,b,c,d will be provided to you. you need to pice only one option from it. 
    Display the option number and name only. After that by giving two lines space give explanation in 100 words. 
    Follow the prompt as it is and must do all the work accordingly.
    Must not give heading Option: and Explanation

    <context>
    {context}
    </context>

    Question: {input}
    """)
    output_parsers = StrOutputParser()

    loader = PyPDFLoader('Info.pdf')
    pages = loader.load_and_split()

    chain = prompt | llm | output_parsers

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(pages)
    db = Chroma.from_documents(documents,OpenAIEmbeddings())

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrival_chain = create_retrieval_chain(retriever,document_chain)

    answer = retrival_chain.invoke({"input":question})

    return answer['answer']

def get_gemini_response(prompt,image):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([image[0],prompt])
    print(response.text)
    return get_response(str(response.text))

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


##initialize our streamlit app

st.set_page_config(page_title="Traditional Knowledge Quiz")

st.header("Quiz Traditional Knowledge")
# input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Generate Answer")

input_prompt = """
                you must have to extract full text from the image and give the text as it is without any changes.
               """

## If ask button is clicked

if submit:
    image_data = input_image_setup(uploaded_file)
    answer=get_gemini_response(input_prompt,image_data)
    st.subheader("The Answer is")
    st.write(answer)
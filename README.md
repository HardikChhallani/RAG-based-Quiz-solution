![image](https://github.com/user-attachments/assets/07c52c23-1c9c-406b-aff8-a62c71ab230c)

# I need you generate a markdown for README.md in git hub of rag based quiz solution app.

source code-
from dotenv import load_dotenv \#Library for loading the Gemini api key
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
\# Check if a file has been uploaded
if uploaded_file is not None:
\# Read the file into bytes
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
    \#\#initialize our streamlit app

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
---

```markdown
# Traditional Knowledge Quiz Solution App

This repository contains a **Streamlit-based application** designed to create a quiz solution app that leverages advanced AI models for analyzing images and generating answers related to traditional knowledge. The app integrates **OpenAI**, **Google Gemini API**, and **LangChain** for seamless functionality.

---

## Features

- **Image Upload**: Upload images in JPG, JPEG, or PNG formats for text extraction.
- **AI-Powered Answer Generation**: Uses OpenAI's ChatGPT and Google's Gemini API to generate accurate answers.
- **Traditional Knowledge Expertise**: Tailored to provide responses based on traditional knowledge contexts.
- **PDF Document Support**: Processes PDF files containing relevant information using LangChain's PyPDFLoader.
- **Text Splitting and Retrieval**: Employs advanced text splitting and retrieval mechanisms for efficient document handling.

---

## Installation

1. Clone the repository:
```

git clone https://github.com/your_username/traditional-knowledge-quiz.git
cd traditional-knowledge-quiz

```

2. Create a virtual environment and activate it:
```

python -m venv env
source env/bin/activate  \# On Windows: env\Scripts\activate

```

3. Install the required dependencies:
```

pip install -r requirements.txt

```

4. Set up environment variables by creating a `.env` file in the root directory:
```

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

```

---

## Usage

1. Run the Streamlit app:
```

streamlit run app.py

```

2. Open the app in your browser at `http://localhost:8501`.

3. Upload an image file (JPG, JPEG, or PNG) and click **Generate Answer** to get AI-generated responses.

---

## Code Structure

- **`app.py`**: Main Streamlit application file.
- **`requirements.txt`**: List of Python dependencies.
- **Environment Variables**:
- `OPENAI_API_KEY`: API key for OpenAI integration.
- `GOOGLE_API_KEY`: API key for Google Gemini API.

---

## Key Libraries Used

- **Streamlit**: For building the interactive web interface.
- **LangChain**: For document processing and retrieval chains.
- **Pillow (PIL)**: For image handling.
- **Chroma**: Vector database for efficient document retrieval.
- **dotenv**: For managing environment variables.

---

## Workflow

1. **Image Upload**: Users upload an image containing text.
2. **Text Extraction**: The app uses Google's Gemini API to extract text from the image.
3. **Answer Generation**:
- The extracted text is passed to an OpenAI model with a custom prompt designed for traditional knowledge questions.
- A detailed answer with explanation is provided based on the retrieved context.

---

## Example Prompt for AI Model

```

You are an expert in the field of traditional knowledge.
The question with four inputs named as a, b, c, d will be provided to you.
You need to pick only one option from it. Display the option number and name only.
After that, by giving two lines space, provide an explanation in 100 words.

```

---

## Future Enhancements

- Add support for additional file formats like PDFs directly from users.
- Improve UI/UX for better accessibility and user experience.
- Enhance multilingual support for diverse user bases.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributors

Feel free to contribute! Fork this repository, make changes, and submit a pull request.

---
```

<div style="text-align: center">⁂</div>

[^1]: https://pplx-res.cloudinary.com/image/upload/v1740501621/user_uploads/nIZDtrfbiZDyUhK/image.jpg


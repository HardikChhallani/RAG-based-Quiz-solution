![image](https://github.com/user-attachments/assets/07c52c23-1c9c-406b-aff8-a62c71ab230c)

```
# 🌟 Traditional Knowledge Quiz App 🌟

Welcome to the **Traditional Knowledge Quiz App**, a fun and interactive way to explore traditional knowledge using cutting-edge AI technologies like **Retrieval-Augmented Generation (RAG)**! This app combines AI-powered text generation with document retrieval to ensure accurate, relevant, and up-to-date answers.

---

## 🚀 Why This App?

This app leverages **RAG** to enhance the accuracy and relevance of AI-generated answers. Traditional language models often rely on outdated or static training data, but RAG bridges this gap by:

- 📚 Retrieving the latest and most relevant information from external sources.
- ✅ Ensuring factual grounding to reduce inaccuracies or hallucinations.
- 💡 Providing domain-specific insights tailored to user queries.

### ✨ What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI framework that combines traditional information retrieval techniques with generative AI models. It ensures that responses are:

- **Grounded in facts**: By retrieving context from external sources.
- **Up-to-date**: Incorporating the latest information without retraining models.
- **Cost-effective**: Eliminating the need for frequent model retraining.

RAG is ideal for applications like this quiz app, where accuracy and context are crucial!

---

## 🎯 Features

- **Image Upload**: Upload images (JPG, JPEG, PNG) for text extraction.
- **AI-Powered Answers**: Get accurate answers with explanations based on traditional knowledge.
- **Interactive UI**: User-friendly interface built with Streamlit.
- **PDF Support**: Processes documents to retrieve additional context for better answers.

---

## 🛠️ Installation

Follow these steps to set up the app on your local machine:

1. **Clone the Repository**:
git clone https://github.com/HardikChhallani/traditional-knowledge-quiz.git
cd traditional-knowledge-quiz



2. **Set Up a Virtual Environment**:
python -m venv env
source env/bin/activate  \# On Windows: env\Scripts\activate

3. **Install Dependencies**:
pip install -r requirements.txt

4. **Add API Keys**:
Create a `.env` file in the root directory with your API keys:

OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key

5. **Run the App**:
streamlit run app.py

6. Open your browser at `http://localhost:8501` and start quizzing!

---

## 🖼️ How It Works

1. Upload an image containing text (e.g., quiz questions).
2. The app uses Google's Gemini API to extract text from the image.
3. Relevant context is retrieved from a PDF or other documents using RAG.
4. OpenAI's GPT generates an accurate answer with an explanation.

---

## 🎉 Why Use RAG in This App?

RAG enhances generative AI by combining retrieval-based methods with text generation capabilities. Here's why it's perfect for our quiz app:

- 🕵️‍♂️ **Accuracy**: Reduces hallucinations by grounding responses in factual data.
- 🔄 **Dynamic Updates**: Provides up-to-date information without retraining models.
- 💸 **Cost Savings**: Avoids expensive retraining by using external knowledge bases.

---

## 📷 Screenshot

![App Screenshot](image.jpg)

---

## 🌟 Future Enhancements

- Add support for multilingual queries.
- Enable direct PDF uploads for more context-rich answers.
- Improve UI/UX for better accessibility.

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Open a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

### Made with ❤️ by AI enthusiasts! Let's make learning fun and engaging! 🎓✨
```

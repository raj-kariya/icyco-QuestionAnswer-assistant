# Icyco
Meet Icyco, your friendly AI assistant ready to scoop up answers to all your Icyco ice cream queries! 🍦 This console-based AI-powered Question Answering system uses Retrieval-Augmented Generation (RAG) to provide accurate information on flavors, services, events, policies, and more, all based on Icyco's internal documents. Powered by Google Generative AI, LangChain, and Pinecone for fast and relevant results.

# High Level Design Overview of AI Assistant 
![High_Level_Design_Assistant](https://github.com/user-attachments/assets/b682b835-522a-402b-8ed7-fa5c70045c23)

# QA assistant for Icyco

Icyco QA Assistant is an console based AI-powered Question Answering system built for the fictional ice cream shop . It uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on company documents like policies, product information, and FAQs.

---

## 🧠 Features

- 💬 Ask questions about Icyco’s products, pricing, policies, etc.
- 📄 Ingests and processes internal documents (PDFs)
- 🔍 Retrieves relevant information using semantic search
- 🤖 Generates answers using Google Generative AI
- ⚡ Fast and responsive user experience with LangChain and Pinecone

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Google Generative AI](https://ai.google.dev/)
- [Python](https://www.python.org/)

---

## 🚀 Getting Started

### 1. Fork and Clone the repo

```bash
git clone https://github.com/your-username/icyco-qa-assistant.git
cd icyco-qa-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file similar to `.env.example` and provide necessary details.

### 4. Run the QA app

````bash
python app.py
````

<!-- Checkout my blog **[Getting Started with RAG by Building a QA Assistant](https://medium.com/@dharshib.8/getting-started-with-rag-by-building-a-qa-assistant-a72b9140b554)**, for more detail explanation of the implementation. -->

---

## Demo


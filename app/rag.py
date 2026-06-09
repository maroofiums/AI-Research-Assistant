from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embedding_model = MistralAIEmbeddings()

model = ChatMistralAI()

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Your task is to answer the user's question using ONLY the provided context.

Rules:
1. Use only the information present in the context.
2. Do not make up facts.
3. If the answer is not available in the context, say:
   "I could not find this information inside the document."
4. Keep answers clear and concise.

Context:
{context}

Question:
{question}
"""
)

db = Chroma(
    "vectorstore",
    embedding_function=embedding_model,
)

def retrieve(question):

    retriever = db.as_retriever(
        search_kwargs={"k":3}
    )

    docs = retriever.invoke(question)

    return docs

def generate_answer(question: str):
    docs = retrieve(question)

    context = "\n\n".join(
        docs.page_content   
        for doc in docs
    )

    chain = prompt | model

    res = chain.invoke(
        {
            "context":context,
            "question":question
        }   
    )

    return {
        "answer": res.content,
        "sources": [
            {
                "page": doc.metadata.get("page"),
                "source": doc.metadata.get("source")
            }
            for doc in docs
        ]
    }


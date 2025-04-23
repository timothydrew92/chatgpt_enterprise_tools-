from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

# Load and embed documents
documents = TextLoader("./data/internal_docs.txt").load()
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embedding_model)

# Setup RAG pipeline
llm = ChatOpenAI(model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
query = "What is our onboarding process for new hires?"
response = qa_chain.run(query)
print(response)

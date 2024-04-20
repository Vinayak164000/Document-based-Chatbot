import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from document_loader import extract_document
import os
os.environ.get('OPENAI_API_KEY')
db_key = os.environ.get('Pinecone_API_KEY')


def load_document():
    directory = './document'
    docs = extract_document(directory)
    return docs

class Model:

    def __init__(self):
        pass    
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key="d5c10780-06c1-45cd-ba30-7f9f5d1e9c11",
        environment="asia-southeast1-gcp-free"
    )
    index_name = "docqna"
    docs = load_document()
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    model_name = "gpt-3.5-turbo"
    llm = OpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")


    def get_similiar_docs(self, query, k=2, score=False):
        if score:
            similar_docs = self.index.similarity_search_with_score(query, k=k)
        else:
            similar_docs = self.index.similarity_search(query, k=k)
        return similar_docs

    def get_answer(self, query):
        chain = load_qa_chain(self.llm, chain_type="stuff")
        similar_docs = self.get_similiar_docs(query)
        answer = chain.run(input_documents=similar_docs, question=query)
        return answer
    
# def module():
#     Model.load_model()
#     return Model.get_answer()


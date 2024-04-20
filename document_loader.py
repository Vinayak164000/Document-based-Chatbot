from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

directory = './document'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def extract_document(directory):
    documents = load_docs(directory)
    docs = split_docs(documents)

    return docs

if __name__ == '__main__':
    documents = load_docs(directory)
    print(len(documents))
    docs = split_docs(documents)
    print(len(docs))

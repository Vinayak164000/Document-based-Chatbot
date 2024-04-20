import os
from llm_langchain import Model

if __name__ == "__main__":
    while True:
        query = input("Enter the query: ")
        answer = Model().get_answer(query)
        print(answer)

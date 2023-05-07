# Import necessary libraries
import os
import textwrap
import pandas as pd
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer

def wrap_text_preserve_newlines(text, width=110):
    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def split_into_chunks(text, tokenizer, max_tokens=500):
    tokens = tokenizer.encode(text, return_tensors="pt").squeeze()
    token_chunks = []

    current_chunk = []
    current_chunk_len = 0
    for token in tokens:
        token_len = len(tokenizer.decode(token.item()))
        if current_chunk_len + token_len + 1 > max_tokens:
            token_chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []
            current_chunk_len = 0
        current_chunk.append(token.item())
        current_chunk_len += token_len + 1

    if current_chunk:
        token_chunks.append(tokenizer.decode(current_chunk))

    return token_chunks

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")

class TextDocument:
    def __init__(self, content, id, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}
        self.metadata['id'] = id

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

data_frame = pd.read_csv("dataset.tsv", sep="\t", nrows=1000)
data = data_frame.to_dict(orient="records")
documents = [TextDocument(content=str(item["answer"]), id=item["id"]) for item in data]
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.75, "max_length": 2048})
chain = load_qa_chain(llm, chain_type="refine")

def truncate_answer(answer, question, tokenizer, max_total_tokens=1000):
    special_tokens = 2
    question_tokens = len(tokenizer.encode(question, return_tensors="pt").squeeze())
    max_answer_tokens = max_total_tokens - question_tokens - special_tokens
    answer_tokens = tokenizer.encode(answer, return_tensors="pt").squeeze()
    truncated_answer = tokenizer.decode(answer_tokens[:max_answer_tokens])
    return truncated_answer

def combined_length_exceeds_limit(question, answer, tokenizer, model_token_limit=1024):
    special_tokens = 2
    question_tokens = len(tokenizer.encode(question, return_tensors="pt").squeeze())
    answer_tokens = len(tokenizer.encode(answer, return_tensors="pt").squeeze())
    return question_tokens + answer_tokens > (model_token_limit - special_tokens)

def process_question(query):
    answers = []

    docs = db.similarity_search(query)
    most_similar_doc = docs[0]
    print(f"Most similar answer: \n{wrap_text_preserve_newlines(str(most_similar_doc.page_content))}\n")

    query_chunks = split_into_chunks(query, tokenizer, max_tokens=500)

    for query_chunk in query_chunks:
        if combined_length_exceeds_limit(query_chunk, str(docs[0].page_content), tokenizer):
            print("The combined length of the question and answer exceeds the model's token limit.")
        else:
            truncated_answer = truncate_answer(str(docs[0].page_content), query_chunk, tokenizer, max_total_tokens=500)
            result = chain.run(input_documents=[TextDocument(content=truncated_answer, id=docs[0].metadata['id'])], question=query_chunk)
            answers.append(result)

    return answers

from flask import Flask, request, jsonify
from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from transformers import TFT5ForConditionalGeneration, T5Tokenizer, pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
import requests
import uuid
import os
import json

import re

app = Flask(__name__)

# Initialize vector_db globally
vector_db = None

# Define Model
local_model = "phi3:mini"
llm = Ollama(
    model=local_model,
    temperature=0.3, 
    top_p=0.5,        
    top_k=20
)

# Function to load and process PDFs
def load_and_process_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    return chunks

# Function to load and process online PDFs
def load_and_process_online_pdf(url):
    print("URL INI")
    print(url)

    # Buat folder pdf_data jika belum ada
    if not os.path.exists('pdf_data'):
        os.makedirs('pdf_data')

    generated_uuid = uuid.uuid4()

    response = requests.get(url)
    with open(f'pdf_data/{generated_uuid}.pdf', 'wb') as file:
        file.write(response.content)

    # loader = OnlinePDFLoader(url)
    loader = UnstructuredPDFLoader(file_path=f'pdf_data/{generated_uuid}.pdf')
    data = loader.load()

    print("DATA NI BOS")
    print(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    return chunks

def youtube_get_context(video_url):

    video_id = video_url.split("=")[1]
    print("PRINT YOUTUBE BOS: ")
    print(video_url)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([d["text"] for d in transcript]) 

    return text

def preprocess_quiz(data):
    print("Preprocessing Quiz...")

    data = data.replace("**", "")
    
    # Use regex to split data into questions and answer keys
    pattern = re.compile(r'(Question \d+:.*?Answer Key: .*?)(?=Question \d+:|$)', re.DOTALL)
    matches = pattern.findall(data.strip())

    output = {
        "questions": []
    }

    for match in matches:
        lines = match.strip().split('\n')
        question_text = lines[0].split(": ", 1)[1].strip()

        choices = {}
        answer_key_line = None

        for line in lines[1:]:
            if re.match(r'^[a-z]\)', line):
                key, value = line.split(')', 1)
                choices[key.strip()] = value.strip()
            elif line.startswith("Answer Key:"):
                answer_key_line = line
                break

        if answer_key_line is None:
            print(f"Error: Answer key not found for question: {question_text}")
            continue

        answer_key = answer_key_line.split("Answer Key: ", 1)[1].split(") ")[0]

        output["questions"].append({
            "question": question_text,
            "choices": choices,
            "answer_key": answer_key
        })

    json_output = json.dumps(output, indent=4)
    print(json_output)
    return json_output

def generate_quiz(material):
    print("generating quiz..")

    # Define the prompt template
    question_prompt = PromptTemplate(
        input_variables=["material"],
        template = """
        Given the following material.
        Material:
        {material}

        ====================================================

        LIST 5 MULTIPLE QUESTIONS.

        The QUESTIONS HAVE TO BE IN Format:
        Question 1 : _____
        a) ___
        b) ___
        c) ___
        d) ___
        Answer Key:

        Question 2 :
        a) ___
        b) ___
        c) ___
        d) ___
        Answer Key:

        Question 3 :
        a) ___
        b) ___
        c) ___
        d) ___
        Answer Key:

        Question 4 :
        a) ___
        b) ___
        c) ___
        d) ___
        Answer Key:

        Question 5 :
        a) ___
        b) ___
        c) ___
        d) ___
        Answer Key:

        """
    )
    chain = question_prompt | llm | StrOutputParser()
    response = chain.invoke({"material": material})

    print("ORIGINAL RESPONSE")
    print(response)

    json_output = preprocess_quiz(response)
    
    return json_output


@app.route('/query', methods=['POST'])
def query():
    print("Start querying")
    global vector_db

    data = request.json
    question = data.get('topic', '')
    pdf_urls = data.get('pdf_urls', [])
    youtube_urls = data.get('youtube_urls', [])

    all_chunks = []

    if pdf_urls or youtube_urls:
        for url in pdf_urls:
            all_chunks.extend(load_and_process_online_pdf(url))
        
        youtube_summaries = []
        for url in youtube_urls:
            youtube_summaries.append(youtube_get_context(url))

        print("PRINT NI BOS: ")
        print(youtube_summaries)
        
        # Add summarized YouTube content as a single document
        youtube_doc_content  = "\n".join(youtube_summaries)
        
        # Membuat youtube_doc dalam format Document
        youtube_doc = [Document(page_content=youtube_doc_content, metadata={})]

        # Menggunakan text_splitter pada youtube_doc
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        youtube_chunks = text_splitter.split_documents(youtube_doc)

        all_chunks.extend(youtube_chunks)

        # Add to vector database
        vector_db = Chroma.from_documents(
            documents=all_chunks, 
            embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
            collection_name="local-rag"
        )

    if vector_db:
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        template = """ 
        Topic(Must Have): {question}
        additional resource: {context}
        Format: Markdown

        Explain, with comprehensive detail and example of '{question}' in 2000 words, use proper markdown format (Title, Heading, etc). 
        Prioritize to generate the Explanation on {question}. 
        include additional resource ONLY if related to the Topic.
        Dont provide Table of Contents

        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        out = chain.invoke(question)
    else:
        # Query without additional context
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Explain, with comprehensive detail and example of '{question}' in 2000 words, use proper markdown format (Title, Heading, etc). 
            Dont provide Table of Contents""",
        )

        chain = prompt | llm | StrOutputParser()
        out = chain.invoke({"question": question})


    # Clean up LaTeX commands
    cleaned_text = re.sub(r'\\begin\{itemize\}|\\end\{itemize\}|\\begin\{align\*\}|\\end\{align\*\}|\\begin\{align\}', '', out)
    cleaned_text = re.sub(r'\\item', '-', cleaned_text)
    cleaned_text = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned_text)
    cleaned_text = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', cleaned_text)
    cleaned_text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', cleaned_text)
    cleaned_text = re.sub(r'\\\\', '', cleaned_text)
    cleaned_text = re.sub(r'\\,', '', cleaned_text)

    ## Generate quiz & Final JSON
    quiz = generate_quiz(cleaned_text)
    
    # final_output = {
    #     "material": cleaned_text,
    #     "questions": json.loads(quiz)
    # }

    final_output = {
        "material": cleaned_text,
        "questions": json.loads(quiz)
    }

    print("FINAL OUTPUT")
    print(json.dumps(final_output, indent=4))

    # return json.dumps(final_output)
    return final_output



if __name__ == '__main__':
    app.run(debug=True)

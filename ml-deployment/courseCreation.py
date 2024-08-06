from flask import Flask, request, jsonify
import requests
import uuid
import os
import re
from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from torch.cuda.amp import autocast, GradScaler
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from deep_translator import GoogleTranslator



scaler = GradScaler()

app = Flask(__name__)

# Initialize vector_db globally
vector_db = None

# Load the fine-tuned model for material & quiz generation
materialQuiz_model, materialQuiz_tokenizer = FastLanguageModel.from_pretrained(
    model_name="/mnt/d/Fauzan/projects/AIC_Development/models/lora_phi3_materialQuizGen",  # FINE-TUNED MODEL
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

materialQuiz_tokenizer = get_chat_template(
    materialQuiz_tokenizer,
    chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
)

local_model = "phi3:latest"
llm = ChatOllama(model=local_model)

# Context Length Encoder Model
from transformers import AutoModel
nomic_embed_text = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

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
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([d["text"] for d in transcript]) 

    return text

def preprocess_quiz(data, lang):
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

        choices = []
        answer_key_line = None
        question_text = lines[0]

        for line in lines[1:]:
            if re.match(r'^[a-z]\)', line):
                key, value = line.split(')', 1)
                choices.append({"letter": key.strip().upper(), "answer": value.strip()})
            elif line.startswith("Answer Key:"):
                answer_key_line = line
                break

        if answer_key_line is None:
            print(f"Error: Answer key not found for question: {question_text}")
        else:
            answer_key = answer_key_line.split("Answer Key: ", 1)[1].strip().upper()

            # Translate output text to Indonesia
            if lang == ('id'):
                question_text = translate_to_indonesia(question_text)
                answer_key = translate_to_indonesia(answer_key)
                for choice in choices:
                    choice['answer'] = translate_to_indonesia(choice['answer'])

            output["questions"].append({
                "question": question_text,
                "choices": choices,
                "key": answer_key[0]
            })

    return output["questions"]

def generate_quiz(material, lang):
    print("generating quiz..")
    print(material)

    # Define the prompt template
    prompt = f"""
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

    FastLanguageModel.for_inference(materialQuiz_model)  # Enable native 2x faster inference

    messages = [
        {"from": "human", "value": prompt}
    ]

    inputs = materialQuiz_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    with autocast():
        outputs = materialQuiz_model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True)

    # outputs = materialQuiz_model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True)
    decoded_output = materialQuiz_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract the part of the output that follows the user prompt
    user_prompt = messages[0]["value"]
    output_text = decoded_output[0].split(user_prompt, 1)[-1].strip()

    json_output = preprocess_quiz(output_text, lang)
    
    return json_output

def translate_to_indonesia(text):
    def split_text(text, limit):
        """
        Memecah teks menjadi bagian yang lebih kecil dengan batas karakter tertentu.
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= limit:
                current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    chunks = split_text(text, 500)

    translator = GoogleTranslator(source='auto', target='id')

    translated_chunks = []
    for chunk in chunks:
        translated_chunk = translator.translate(chunk)
        translated_chunks.append(translated_chunk)

    translated_text = ' '.join(translated_chunks)
    return translated_text


@app.route('/query', methods=['POST'])
def query():
    print("Start querying")
    global vector_db

    # Extract Data from request
    data = request.json
    courseName = data.get('name', '')
    description = data.get('description', '')
    chapters = data.get('content', [])
    pdf_urls = data.get('pdf_urls', [])
    youtube_urls = data.get('youtube_urls', [])
    lang = data.get('lang', '')
    course_content = []

    for chapter in chapters:

        print("CHAPTER:")
        print(chapter)

        chapter_number = chapter['chapter']
        chapter_title = chapter['title']
        topicPrompt = f"Topic: {courseName} - {chapter_title}"

        all_chunks = []
        if pdf_urls or youtube_urls:
            for url in pdf_urls:
                all_chunks.extend(load_and_process_online_pdf(url))
            
            youtube_summaries = []
            for url in youtube_urls:
                youtube_summaries.append(youtube_get_context(url))
            
            # Add summarized YouTube content as a single document
            youtube_doc_content  = "\n".join(youtube_summaries)
            
            # Membuat youtube_doc dalam format Document
            youtube_doc = [Document(page_content=youtube_doc_content, metadata={})]

            # Menggunakan text_splitter pada youtube_doc
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            youtube_chunks = text_splitter.split_documents(youtube_doc)

            all_chunks.extend(youtube_chunks)
            retriever = FAISS.from_documents(all_chunks, OllamaEmbeddings(model="nomic-embed-text", show_progress=True)).as_retriever()

        if all_chunks:

            compressor = LLMChainExtractor.from_llm(llm)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            
            context_template = f"""Generate a short (200 words) summary of topic: {topicPrompt}"""

            compressed_docs = compression_retriever.invoke(context_template)

            context = "\n\n".join(doc.page_content for doc in compressed_docs)

            if len(context) > 150:
                context = context[:150]

            prompt = f""" 
            Topic(Must Have): {topicPrompt}
            additional resource: {context}
            Format: Markdown

            Explain, with comprehensive detail and example of '{topicPrompt}' in 2000 words, use proper markdown format (Title, Heading, etc). 
            Prioritize to generate the Explanation on {topicPrompt}. 
            include additional resource ONLY if related to the Topic.
            Dont provide Table of Contents

            """

            FastLanguageModel.for_inference(materialQuiz_model)  # Enable native 2x faster inference

            messages = [
                {"from": "human", "value": prompt}
            ]

            inputs = materialQuiz_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
            ).to("cuda")

            with autocast():
                outputs = materialQuiz_model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True)

            decoded_output = materialQuiz_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract the part of the output that follows the user prompt
            user_prompt = messages[0]["value"]
            output_text = decoded_output[0].split(user_prompt, 1)[-1].strip()

            # Generate quiz & Final JSON
            quiz = generate_quiz(topicPrompt)

            course_content.append({
                "chapter": chapter_number,
                "title": chapter_title,
                "text": output_text,
                "quiz": quiz
            })
            
        else:
            prompt = f"""Explain, with comprehensive detail and example of '{topicPrompt}' in 2000 words, use proper markdown format (Title, Heading, etc). Dont provide Table of Contents"""
            
            FastLanguageModel.for_inference(materialQuiz_model)  # Enable native 2x faster inference

            messages = [
                {"from": "human", "value": prompt}
            ]

            inputs = materialQuiz_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Must add for generation
                return_tensors="pt",
            ).to("cuda")

            with autocast():
                outputs = materialQuiz_model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True)

            decoded_output = materialQuiz_tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Extract the part of the output that follows the user prompt
            user_prompt = messages[0]["value"]
            output_text = decoded_output[0].split(user_prompt, 1)[-1].strip()

            # Generate quiz & Final JSON
            quiz = generate_quiz(topicPrompt, lang)

            # Translate output text to Indonesia
            if lang == ('id'):
                output_text = translate_to_indonesia(output_text)

            # course_content.append({
            #     "chapter": chapter_number,
            #     "title": chapter_title,
            #     "text": output_text,
            #     "quiz": quiz
            # })

            course_content.append({
                "chapter": chapter_number,
                "title": chapter_title,
                "text": output_text,
                "quiz": quiz
            })

        
    final_output = {
        "name": courseName,
        "description": description,
        "pdf_urls": pdf_urls,
        "youtube_urls": youtube_urls,
        "content": course_content
    }

    # Translate to Indo


    # return json.dumps(final_output)
    return jsonify(final_output)


if __name__ == '__main__':
    app.run(debug=True)

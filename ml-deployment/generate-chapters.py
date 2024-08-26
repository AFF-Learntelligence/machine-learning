from flask import Flask, request, jsonify
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from deep_translator import GoogleTranslator

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

app = Flask(__name__)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/mnt/d/Fauzan/projects/AIC_Development/models/lora_phi3_chapterGen", # FINE-TUNED MODEL
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

tokenizer = get_chat_template(
        tokenizer,
        chat_template="phi-3",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
    )

# Post Endpoint
@app.route("/generate-chapters", methods=["POST"])
def generate_chapters():
    print("Post /generate-chapters called")
    json_content = request.json
    course_title = json_content.get("course_title")
    course_length = json_content.get("course_length")
    lang = json_content.get('lang', '')
    
    
    prompt = f""""
    List {course_length} chapter name for a course '{course_title}'. 
    Write just the outline name only! no need for explanation For example: Chapter 1: ___ Chapter 2: ___ Chapter 3: ___
    """

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    messages = [
        {"from": "human", "value": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(input_ids=inputs, max_new_tokens=2000, use_cache=True)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract the part of the output that follows the user prompt
    user_prompt = messages[0]["value"]
    output_text = decoded_output[0].split(user_prompt, 1)[-1].strip()

    print(output_text)
    
    # Memisahkan string berdasarkan kata "Chapter"
    chapters = output_text.strip().split("Chapter")
    chapters = [chapter.strip() for chapter in chapters if chapter]  # Menghapus spasi dan elemen kosong

    # Menambahkan kembali kata "Chapter" di depan setiap elemen
    chapters = [f"Chapter {chapter}" for chapter in chapters]
    
    # Format the output to the required structure

    if lang == ('id'):
        formatted_chapters = [
            {"chapter": int(chapter.split(":")[0].split()[1]), "title": translate_to_indonesia(chapter.split(":")[1].strip())}
            for chapter in chapters
        ]
    
    else:
        formatted_chapters = [
            {"chapter": int(chapter.split(":")[0].split()[1]), "title": chapter.split(":")[1].strip()}
            for chapter in chapters
        ]

    

    response_answer = {"data": formatted_chapters}
    return jsonify(response_answer)


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()

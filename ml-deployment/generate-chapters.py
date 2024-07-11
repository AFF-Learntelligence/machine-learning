from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)
# llm = Ollama(model='llama3')
llm = Ollama(model='phi3:mini')

# Post Endpoint
@app.route("/generate-chapters", methods=["POST"])
def generate_chapters():
    print("Post /generate-chapters called")
    json_content = request.json
    course_title = json_content.get("course_title")
    course_length = json_content.get("course_length")
    
    prompt = f""""
    List {course_length} chapter name for a course '{course_title}'. 
    Write just the outline name only! no need for explanation For example: Chapter 1: ___ Chapter 2: ___ Chapter 3: ___
    """

    response = llm.invoke(prompt)
    
    # Memisahkan string berdasarkan kata "Chapter"
    chapters = response.strip().split("Chapter")
    chapters = [chapter.strip() for chapter in chapters if chapter]  # Menghapus spasi dan elemen kosong

    # Menambahkan kembali kata "Chapter" di depan setiap elemen
    chapters = [f"Chapter {chapter}" for chapter in chapters]
    
    response_answer = {"data": chapters}
    return jsonify(response_answer)


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()

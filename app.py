from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from flask import Flask, render_template, request
import markdown
import torch

app = Flask(__name__)

api_key = "AIzaSyCjdyDhgzzb90DwvSFPTt2wlU5yNyyFmew"

# Use GPU acceleration if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_relevant_context(query):
    # Get relevant context from vector store
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                               model_kwargs={'device': device})
    vector_db = Chroma(persist_directory="./chroma_db_nccn1", embedding_function=embedding_function)
    search_result = vector_db.similarity_search(query, k=6)
    for result in search_result:
        context += result.page_content + "\n"
    return context


def rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (f"""
    Provide a detailed and technical answer to the question: '{query}'.
    Please include all relevant information and specifications with time.
    Provide a nutritional details.
    Provide the recipe name.
    CONTEXT: '{context}'

     Answer:
    """).format(query=query, context=context)
    return prompt


def generate_ans(prompt):
    # Generate answer using generative model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["prompt"]
        print(query)

        context = get_relevant_context(query)
        prompt = rag_prompt(query, context)
        answer = generate_ans(prompt)
        html_response = markdown.markdown(answer)

        return render_template("index.html", response=html_response)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, jsonify
import fitz  # PyMuPDF
import os
from rag import init_llm, process_pdfs, get_answer

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
# Add these as global variables after app initialization
llm = init_llm()
retriever = None

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/conversation', methods=['GET', 'POST'])
def conversation():
    global retriever
    
    if request.method == 'GET':
        return render_template('conversation.html')
    
    if request.method == 'POST':
        question = request.form.get('question')
        
        # Initialize retriever if not already done
        if retriever is None:
            file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) 
                         for f in os.listdir(app.config['UPLOAD_FOLDER'])]
            retriever = process_pdfs(file_paths)
        
        # Get answer and source
        answer, is_web_search = get_answer(question, retriever, llm)
        
        return jsonify({
            'answer': answer,
            'isWebSearch': is_web_search
        })
    

@app.route('/all_files')
def all_files():
    files = list()
    for file in os.listdir(UPLOAD_FOLDER):
        files.append(file)
    return render_template('all_files.html', files=files)

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return 'No files part in the request', 400

    files = request.files.getlist('files')
    extracted_texts = []

    for file in files:
        if file.filename == '':
            return 'One or more files have no filename', 400
        if file and file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            text = extract_text_from_pdf(file_path)
            extracted_texts.append({'filename': file.filename, 'text': text})
        else:
            return f'File {file.filename} is not a PDF', 400

    return render_template('display_texts.html', extracted_texts=extracted_texts)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        document = fitz.open(pdf_path)
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        document.close()
    except Exception as e:
        text = f"An error occurred while processing {pdf_path}: {str(e)}"
    return text


if __name__ == "__main__":
    app.run(debug=True)

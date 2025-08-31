from flask import Flask, request, jsonify, send_file
import json
import re
import os
import requests
import mammoth
import PyPDF2
import io
import pytesseract
from PIL import Image
from docx import Document
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)

CORS(app=app)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    job_title = data.get("jobTitle")
    job_description = data.get("jobDescription")
    extracted_text = data.get("extractedText")

    if not all([job_title, job_description, extracted_text]):
        return jsonify({"error": "Missing fields"}), 400

    prompt = f"""
        You are an AI career assistant. You will receive:
        1. A job title (plain text).
        2. A job description (plain text).
        3. A resume's extracted text.

        Your task is to:
        - Compare the resume to the job description.
        - Decide if the candidate should apply: respond with "Yes" or "No" in the "decision" field.
        - Write the "reason" as exactly **one sentence** that speaks directly to the candidate (use "you", not third person).
        - Provide a "score" (0–100) indicating how well the resume matches the job description.
        - If "decision" = "Yes" **and** score ≥ 60:
            - Generate a "coverLetter" that:
                1. Has a professional, personalized, and concise tone.
                2. Is 3–4 paragraphs long.
                3. Starts with a title like: "Application for <jobTitle>".
                4. Is tailored to the job description and highlights the candidate's relevant skills and experience.
        - If "decision" = "No" or score < 60, set "coverLetter" to an empty string "".
        - If "decision" = "Yes", generate "resumeEnhancements" with clear, specific, and actionable suggestions to make the resume a perfect match for the job description.
            - Highlight missing or weak skills, keywords, tools, certifications, and responsibilities from the job description.
            - Suggest clear edits: add, rewrite, or reorder resume bullet points to match the role.
            - Recommend metrics only when directly tied to listed experiences.
            - Avoid generic tips; link every suggestion to actual resume content.
            - Keep advice specific and contextual to the provided resume and job description.
        - Always speak directly to the user (e.g., "You have strong experience..." not "The candidate has...").

        Respond **only** in the following JSON format with no extra text, markdown, or explanations:

        {{
            "decision": "Yes" or "No",
            "reason": "<one-sentence reason>",
            "score": <integer from 0 to 100>,
            "coverLetter": "<cover letter text or empty string>",
            "resumeEnhancements": "<specific resume suggestions to improve ATS match or empty string>",
            "jobSummary": "<brief summary of the job description including key skills and requirements>"
        }}

        Job Title:
        {job_title}

        Job Description:
        {job_description}

        Resume:
        {extracted_text}
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    res = requests.post(url, json=payload)
    if res.status_code != 200:
        return jsonify({"error": res.json()}), res.status_code

    text_output = res.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()

    # Strip triple backticks if present
    if text_output.startswith("```"):
        text_output = re.sub(r"^```json", "", text_output, flags=re.IGNORECASE)
        text_output = re.sub(r"^```", "", text_output)
        text_output = re.sub(r"```$", "", text_output)
        text_output = text_output.strip()

    try:
        parsed = json.loads(text_output)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON from Gemini", "raw": text_output}), 500

    return jsonify(parsed)

@app.route("/extract/text", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        print("No file part in the request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename.lower()
    file_bytes = file.read()

    try:
        # Handle PDF
        if filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

            # If no text found, fallback to OCR
            if not text.strip():
                text = ocr_pdf(file_bytes)

            return jsonify({"text": text.strip()})

        # Handle DOCX
        elif filename.endswith(".docx"):
            result = mammoth.extract_raw_text(io.BytesIO(file_bytes))
            return jsonify({"text": result.value.strip()})

        # Handle TXT
        elif filename.endswith(".txt"):
            return jsonify({"text": file_bytes.decode("utf-8").strip()})

        # Handle Images (JPG, PNG, etc.)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            text = ocr_image(file_bytes)
            return jsonify({"text": text.strip()})

        else:
            print("Unsupported file type")
            return jsonify({"error": "Unsupported file type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/text/to/docx", methods=["POST"])
def text_to_docx():
    data = request.get_json()
    if not data or "text" not in data:
        return {"error": "No text provided"}, 400

    text = data["text"]

    # Create DOCX
    doc = Document()

    # Split on double line breaks first, fall back to single
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    for para in paragraphs:
        doc.add_paragraph(para)

    # Save to memory
    output = io.BytesIO()
    doc.save(output)
    output.seek(0)

    # Send file to client with proper headers
    return send_file(
        output,
        as_attachment=True,
        download_name="cover_letter.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def ocr_pdf(file_bytes):
    """Convert PDF pages to images and run OCR"""
    from pdf2image import convert_from_bytes
    images = convert_from_bytes(file_bytes)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text


def ocr_image(file_bytes):
    """Run OCR on image file"""
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
# AI-Powered Medical Report Assistant - Gemini Pro Version
# Streamlit + OCR + Generative AI + PDF Export + Lifestyle Suggestions

import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import cv2
import tempfile
import re
import pandas as pd
from fpdf import FPDF
import base64
import google.generativeai as genai

# Set your Gemini API key here
genai.configure(api_key="AIzaSyDU1jsQM5lIijH04xI7sRGBuk4j5u7KbkE")
model = genai.GenerativeModel("gemini-pro")

st.set_page_config(page_title="Medical Report AI Assistant", layout="wide")
st.title("🩺 Medical Report AI Assistant")

# Helper: Preprocess image for OCR (simplified)
def preprocess_image(img):
    # Convert to grayscale (no blur or threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Optional: you can uncomment threshold if needed
    # _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # return binary
    return gray

# Helper: Extract text from image using Tesseract with config
def extract_text_from_image(image):
    img_np = np.array(image)
    preprocessed = preprocess_image(img_np)
    # Use Tesseract with Page Segmentation Mode 6 (Assume a single uniform block of text)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(preprocessed, config=custom_config)

# Extract images from PDF (high DPI for better quality)
@st.cache_data
def extract_images_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    for page in doc:
        # Render page to image with 300 dpi for better clarity
        zoom = 300 / 72  # 72 dpi is default, scale up to 300 dpi
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)  # no alpha for RGB
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

# Parse lab results into structured data
def parse_lab_results(text):
    pattern = r"(?P<Test>[A-Za-z\s\-/]+)\s+(?P<Value>\d+\.?\d*)\s*(?P<Unit>mg/dL|g/dL|mmol/L|%)\s*(?P<Range>\(?\d+\-\d+\)?|\d+\u2013\d+)"
    matches = re.findall(pattern, text)
    data = []
    for match in matches:
        test, value, unit, range_val = match
        try:
            low, high = map(float, re.findall(r"\d+\.?\d*", range_val))
            value = float(value)
            category = "Normal"
            if value < low:
                category = "Low"
            elif value > high:
                category = "High"
            data.append({
                "Test": test.strip(),
                "Value": value,
                "Unit": unit,
                "Normal Range": range_val,
                "Category": category
            })
        except Exception:
            # Skip parsing errors
            continue
    return pd.DataFrame(data)

# Generate explanation using Gemini Pro
@st.cache_data(show_spinner=False)
def generate_explanation(test, value, normal_range):
    prompt = f"""
    Explain in simple language what it means if the patient’s {test} is {value}, given the normal range is {normal_range}.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Generate lifestyle/follow-up suggestions
def generate_suggestions(test, value):
    prompt = f"""
    What lifestyle advice or medical follow-up should be considered for a patient with {test} value of {value}?
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Generate summary PDF
def generate_pdf_report(df, explanations, suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Medical Report AI Summary", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    for i, row in df.iterrows():
        pdf.cell(200, 10, f"{row['Test']} - {row['Value']} {row['Unit']} ({row['Category']})", ln=True)
        pdf.multi_cell(0, 10, f"Explanation: {explanations[i]}")
        pdf.multi_cell(0, 10, f"Suggestions: {suggestions[i]}")
        pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="report_summary.pdf">📄 Download Report PDF</a>'
            return href

# UI - File uploader
uploaded_file = st.file_uploader("📤 Upload medical report (PDF or image)", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pages = extract_images_from_pdf(uploaded_file)
    else:
        pages = [Image.open(uploaded_file)]

    full_text = ""
    for img in pages:
        st.image(img, caption="Uploaded Page", use_column_width=True)
        text = extract_text_from_image(img)
        full_text += text + "\n"

    st.subheader("🔍 Extracted Text")
    st.text_area("OCR Result", value=full_text, height=300)

    lab_df = parse_lab_results(full_text)
    st.subheader("🧾 Parsed Lab Results")
    st.dataframe(lab_df)

    if not lab_df.empty:
        st.subheader("🧠 AI Explanation")
        explanations = []
        suggestions = []

        for idx, row in lab_df.iterrows():
            with st.expander(f"{row['Test']} ({row['Category']})"):
                explanation = generate_explanation(row['Test'], row['Value'], f"{row['Normal Range']} {row['Unit']}")
                suggestion = generate_suggestions(row['Test'], row['Value'])
                explanations.append(explanation)
                suggestions.append(suggestion)

                st.markdown(f"**Explanation:**\n{explanation}")
                st.markdown(f"**📝 Suggested Action:**\n{suggestion}")

        st.subheader("📄 Export Summary")
        download_link = generate_pdf_report(lab_df, explanations, suggestions)
        st.markdown(download_link, unsafe_allow_html=True)

st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
This app extracts data from medical reports, explains test results using AI, and offers follow-up advice.
- Powered by Gemini Pro
- Uses OCR (Tesseract + OpenCV)
- PDF export included
""")

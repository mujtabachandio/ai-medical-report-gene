# AI-Powered Medical Report Assistant - Gemini Pro Version
# Streamlit + OCR + Generative AI + PDF Export + Lifestyle Suggestions

import streamlit as st
import fitz  # PyMuPDF
import pytesseract
import os
from PIL import Image
import numpy as np
import cv2
import tempfile
import re
import pandas as pd
from fpdf import FPDF
import base64
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Set page config
st.set_page_config(page_title="Medical Report AI Assistant", layout="wide")

# Load environment variables
load_dotenv()

# Add custom CSS 
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #1a1a2e;
        padding: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: #e6e6e6;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #e6e6e6;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
    
    /* Premium card styling */
    .premium-card {
        background: linear-gradient(145deg, #16213e, #1a1a2e);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.1);
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Card icon styling */
    .card-icon {
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 1.2em;
    }
    
    /* Animation classes */
    .slide-up {
        animation: slideUp 0.5s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #16213e;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 2px dashed rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #4CAF50;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #16213e !important;
        color: #e6e6e6 !important;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.1);
    }
    
    .dataframe th {
        background-color: #1a1a2e !important;
        color: #4CAF50 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .dataframe td {
        padding: 0.75rem !important;
        border-bottom: 1px solid rgba(76, 175, 80, 0.1) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #16213e !important;
        color: #e6e6e6 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: 1px solid rgba(76, 175, 80, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #1a1a2e !important;
        border-color: #4CAF50 !important;
    }
    
    /* Info box styling */
    .stInfo {
        background: linear-gradient(145deg, #16213e, #1a1a2e) !important;
        border-radius: 15px !important;
        border-left: 5px solid #4CAF50 !important;
        color: #e6e6e6 !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Warning box styling */
    .stWarning {
        background: linear-gradient(145deg, #16213e, #1a1a2e) !important;
        border-radius: 15px !important;
        border-left: 5px solid #FFA500 !important;
        color: #e6e6e6 !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Error box styling */
    .stError {
        background: linear-gradient(145deg, #16213e, #1a1a2e) !important;
        border-radius: 15px !important;
        border-left: 5px solid #FF5252 !important;
        color: #e6e6e6 !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(145deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(145deg, #45a049, #4CAF50);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(145deg, #16213e, #1a1a2e);
        color: #e6e6e6;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(145deg, #16213e, #1a1a2e);
    }
    
    /* Download link styling */
    .download-link {
        display: inline-block;
        background: linear-gradient(145deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        text-align: center;
        margin: 1rem 0;
    }
    
    .download-link:hover {
        background: linear-gradient(145deg, #45a049, #4CAF50);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    /* Custom div styling */
    .custom-div {
        background: linear-gradient(145deg, #16213e, #1a1a2e);
        color: #e6e6e6;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(76, 175, 80, 0.1);
        transition: all 0.3s ease;
    }
    
    .custom-div:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Text colors */
    .text-primary {
        color: #e6e6e6 !important;
    }
    
    .text-secondary {
        color: #b3b3b3 !important;
    }
    
    .text-accent {
        color: #4CAF50 !important;
    }
    
    /* Loading spinner styling */
    .stSpinner {
        color: #4CAF50 !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #4CAF50 !important;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #16213e;
        color: #e6e6e6;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Configure Tesseract path
def configure_tesseract():
    # List of possible Tesseract installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe',
        r'tesseract'  # This will work if tesseract is in PATH
    ]
    
    # Try to find Tesseract
    tesseract_found = False
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            break
    
    if not tesseract_found:
        st.error("""
        Tesseract OCR is not installed or not found. Please follow these steps:
        1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install it with default settings
        3. Make sure to check 'Add to system PATH' during installation
        4. Restart your computer
        5. Run this application again
        """)
        st.stop()

# Configure Tesseract at startup
configure_tesseract()

# Configure Gemini API
def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("""
        Gemini API key not found. Please follow these steps:
        1. Get a free Gemini API key from: https://makersuite.google.com/app/apikey
        2. Create a .env file in the project directory
        3. Add your API key to the .env file:
           GOOGLE_API_KEY=your_api_key_here
        4. Restart the application
        """)
        st.stop()
    
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Use gemini-1.5-flash which is optimized for speed and efficiency
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test the API key with a simple prompt
        response = model.generate_content("Hello")
        if response and hasattr(response, 'text'):
            return model
        else:
            st.error("Failed to initialize Gemini model. Please try again.")
            st.stop()
    except Exception as e:
        if "404" in str(e):
            st.error("""
            Model not found. This could be due to:
            1. API version mismatch
            2. Model name change
            3. Regional availability
            
            Please try again in a few minutes or check the latest model names at:
            https://ai.google.dev/gemini-api/docs/models
            """)
        elif "429" in str(e):
            st.error("""
            Rate limit exceeded. Please wait a few minutes before trying again.
            The free tier has limits on:
            - Requests per minute
            - Requests per day
            - Input tokens per minute
            
            Try these solutions:
            1. Wait 1-2 minutes before trying again
            2. Process one test result at a time
            3. Keep explanations brief
            """)
        else:
            st.error(f"""
            Error configuring Gemini API: {str(e)}
            Please check your API key and try again.
            Make sure you're using the free API key from: https://makersuite.google.com/app/apikey
            """)
        st.stop()

# Configure Gemini at startup
model = configure_gemini()

# Update the main title and layout
st.title("🩺 Medical Report AI Assistant")
st.markdown("""
<div style='text-align: center; color: #b3b3b3; margin-bottom: 2rem; font-size: 1.2rem;'>
    Advanced medical report analysis powered by cutting-edge AI technology
</div>
""", unsafe_allow_html=True)

# Update file uploader section
st.markdown("""
<div class='custom-div'>
    <h3 style='color: #e6e6e6; margin-bottom: 1rem;'>📄 Upload Your Medical Report</h3>
    <p style='color: #b3b3b3; margin-bottom: 1rem;'>Supported formats: PDF, PNG, JPG, JPEG</p>
    <div class='tooltip'>
        <span class='tooltiptext'>Upload your medical report for instant analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Add file uploader with custom styling
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg", "pdf"])

# Helper: Preprocess image for OCR
def preprocess_image(img):
    # Convert to numpy array if it's a PIL Image
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

# Extract text from image using Tesseract
def extract_text_from_image(image):
    try:
        # Preprocess the image
        preprocessed = preprocess_image(image)
        
        # Custom configuration for better accuracy
        custom_config = r'--oem 3 --psm 6'
        
        # Extract text
        text = pytesseract.image_to_string(preprocessed, config=custom_config)
        
        # Clean up the text
        text = text.replace('\n\n', '\n')  # Remove double newlines
        text = re.sub(r'\s+', ' ', text)   # Normalize whitespace
        text = text.strip()                # Remove leading/trailing whitespace
        
        # Split text into lines
        lines = []
        current_line = ""
        
        # Split by common delimiters and test names
        parts = re.split(r'(HEMOGLOBIN|RBC COUNT|BLOOD INDICES|WBC COUNT|DIFFERENTIAL WBC COUNT|PLATELET COUNT|Mean Corpuscular Volume|MCH|MCHC|RDW|Neutrophils|Lymphocytes|Eosinophils|Monocytes|Basophils)', text)
        
        for part in parts:
            if part.strip():
                if re.match(r'HEMOGLOBIN|RBC COUNT|BLOOD INDICES|WBC COUNT|DIFFERENTIAL WBC COUNT|PLATELET COUNT|Mean Corpuscular Volume|MCH|MCHC|RDW|Neutrophils|Lymphocytes|Eosinophils|Monocytes|Basophils', part):
                    if current_line:
                        lines.append(current_line.strip())
                    current_line = part
                else:
                    current_line += " " + part
        
        if current_line:
            lines.append(current_line.strip())
        
        # Join lines back together
        text = '\n'.join(lines)
        
        return text
    except Exception as e:
        st.error(f"Error during text extraction: {str(e)}")
        return ""

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
    data = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    start_parsing = False
    
    for line in lines:
        if "COMPLETE BLOOD COUNT" in line.upper():
            start_parsing = True
            continue
            
        if not start_parsing:
            continue
            
        for test_key, test_data in test_info.items():
            if any(var in line.upper() for var in test_data['variations']):
                patterns = [
                    r'(\d+\.?\d*)\s*(Low|High|Borderline)?\s*(\d+\.?\d*)-(\d+\.?\d*)\s*([a-zA-Z/%]+)',
                    r'(\d+\.?\d*)\s*([a-zA-Z/%]+)\s*(Low|High|Borderline)?\s*(\d+\.?\d*)-(\d+\.?\d*)',
                    r'(\d+\.?\d*)\s*(Low|High|Borderline)?\s*(\d+\.?\d*)-(\d+\.?\d*)'
                ]
                
                value_match = None
                for pattern in patterns:
                    value_match = re.search(pattern, line)
                    if value_match:
                        break
                
                if value_match:
                    if len(value_match.groups()) == 5:
                        if 'Unit' in pattern:
                            value = float(value_match.group(1))
                            unit = value_match.group(2)
                            status = value_match.group(3)
                            min_val = float(value_match.group(4))
                            max_val = float(value_match.group(5))
                        else:
                            value = float(value_match.group(1))
                            status = value_match.group(2)
                            min_val = float(value_match.group(3))
                            max_val = float(value_match.group(4))
                            unit = test_data['unit']
                    else:
                        value = float(value_match.group(1))
                        status = value_match.group(2)
                        min_val = float(value_match.group(3))
                        max_val = float(value_match.group(4))
                        unit = test_data['unit']
                    
                    # Determine category
                    if status:
                        category = status
                    elif value < min_val:
                        category = "Low"
                    elif value > max_val:
                        category = "High"
                    else:
                        category = "Normal"
                    
                    # Add to data if not already present
                    if not any(d['Test'] == test_data['name'] for d in data):
                        data.append({
                            "Test": test_data['name'],
                            "Value": value,
                            "Unit": unit,
                            "Normal Range": f"{min_val}-{max_val}",
                            "Category": category
                        })
                    break
    
    return pd.DataFrame(data)

# Generate explanation using Gemini
@st.cache_data(show_spinner=False)
def generate_explanation(test, value, normal_range):
    try:
        prompt = f"""
        Explain in simple language what it means if the patient's {test} is {value}, given the normal range is {normal_range}.
        Keep the explanation brief and easy to understand.
        """
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            return response.text.strip()
        return "Unable to generate explanation at this time."
    except Exception as e:
        if "404" in str(e):
            st.error("Model not found. Please try again in a few minutes.")
        elif "429" in str(e):
            st.error("Rate limit exceeded. Please wait 1-2 minutes before trying again.")
        else:
            st.error(f"Error generating explanation: {str(e)}")
        return "Unable to generate explanation at this time."

# Generate lifestyle/follow-up suggestions
def generate_suggestions(test, value):
    try:
        prompt = f"""
        What lifestyle advice or medical follow-up should be considered for a patient with {test} value of {value}?
        Keep the suggestions practical and actionable.
        Format each suggestion as a clear action item without using markdown or special characters.
        Start each suggestion with an action verb.
        """
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            # Clean up the response to remove any markdown formatting
            text = response.text.strip()
            # Remove any asterisks or markdown formatting
            text = text.replace('*', '').replace('**', '').replace('__', '')
            # Split into lines and clean each line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Join back with proper spacing
            return ' '.join(lines)
        return "Unable to generate suggestions at this time."
    except Exception as e:
        if "404" in str(e):
            st.error("Model not found. Please try again in a few minutes.")
        elif "429" in str(e):
            st.error("Rate limit exceeded. Please wait 1-2 minutes before trying again.")
        else:
            st.error(f"Error generating suggestions: {str(e)}")
        return "Unable to generate suggestions at this time."

# Add error handling function
def handle_ocr_errors(text):
    if not text:
        st.error("No text could be extracted. Please ensure the image is clear and readable.")
        return False
    return True

# Add risk summary function
def generate_risk_summary(df):
    critical_tests = df[df['Category'].isin(['High', 'Low'])]
    if not critical_tests.empty:
        summary = "Critical Findings:\n"
        for _, row in critical_tests.iterrows():
            summary += f"- {row['Test']} is {row['Category'].lower()}\n"
        return summary
    return "All test results are within normal range."

# Update test_info with more variations
test_info = {
    'HEMOGLOBIN': {'unit': 'g/dL', 'range': '13.0-17.0', 'name': 'Hemoglobin', 'variations': ['HEMOGLOBIN', 'HB', 'HGB', 'HEMOGLOBIN:', 'HB:', 'HGB:']},
    'RBC': {'unit': 'mill/cumm', 'range': '4.5-5.5', 'name': 'Red Blood Cells', 'variations': ['RBC', 'RBCCOUNT', 'RED BLOOD CELLS', 'TOTAL RBC', 'RBC:', 'RBC COUNT:', 'TOTAL RBC:']},
    'PCV': {'unit': '%', 'range': '40-50', 'name': 'Packed Cell Volume', 'variations': ['PCV', 'HCT', 'HEMATOCRIT', 'PACKED CELL VOLUME', 'PCV:', 'HCT:', 'HEMATOCRIT:']},
    'MCV': {'unit': 'fL', 'range': '83-101', 'name': 'Mean Corpuscular Volume', 'variations': ['MCV', 'MEAN CORPUSCULAR VOLUME', 'MCV:', 'MEAN CORPUSCULAR VOLUME:']},
    'MCH': {'unit': 'pg', 'range': '27-32', 'name': 'Mean Corpuscular Hemoglobin', 'variations': ['MCH', 'MEAN CORPUSCULAR HEMOGLOBIN', 'MCH:', 'MEAN CORPUSCULAR HEMOGLOBIN:']},
    'MCHC': {'unit': 'g/dL', 'range': '32.5-34.5', 'name': 'Mean Corpuscular Hemoglobin Concentration', 'variations': ['MCHC', 'MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION', 'MCHC:', 'MEAN CORPUSCULAR HEMOGLOBIN CONCENTRATION:']},
    'RDW': {'unit': '%', 'range': '11.6-14.0', 'name': 'Red Cell Distribution Width', 'variations': ['RDW', 'RED CELL DISTRIBUTION WIDTH', 'RDW:', 'RED CELL DISTRIBUTION WIDTH:']},
    'WBC': {'unit': 'cumm', 'range': '4000-11000', 'name': 'White Blood Cells', 'variations': ['WBC', 'WBCCOUNT', 'LEUCOCYTES', 'WHITE BLOOD CELLS', 'TOTAL WBC', 'WBC:', 'WBC COUNT:', 'TOTAL WBC:']},
    'NEUTROPHILS': {'unit': '%', 'range': '50-62', 'name': 'Neutrophils', 'variations': ['NEUTROPHILS', 'NEUTROPHIL', 'NEUTROPHILS:', 'NEUTROPHIL:']},
    'LYMPHOCYTES': {'unit': '%', 'range': '20-40', 'name': 'Lymphocytes', 'variations': ['LYMPHOCYTES', 'LYMPHOCYTE', 'LYMPHOCYTES:', 'LYMPHOCYTE:']},
    'EOSINOPHILS': {'unit': '%', 'range': '0-6', 'name': 'Eosinophils', 'variations': ['EOSINOPHILS', 'EOSINOPHIL', 'EOSINOPHILS:', 'EOSINOPHIL:']},
    'MONOCYTES': {'unit': '%', 'range': '0-10', 'name': 'Monocytes', 'variations': ['MONOCYTES', 'MONOCYTE', 'MONOCYTES:', 'MONOCYTE:']},
    'BASOPHILS': {'unit': '%', 'range': '0-2', 'name': 'Basophils', 'variations': ['BASOPHILS', 'BASOPHIL', 'BASOPHILS:', 'BASOPHIL:']},
    'PLATELETS': {'unit': 'cumm', 'range': '150000-410000', 'name': 'Platelets', 'variations': ['PLATELETS', 'PLT', 'PLATELET COUNT', 'PLATELETS:', 'PLT:', 'PLATELET COUNT:']}
}

# Update PDF generation with more detailed sections
def generate_pdf_report(df, explanations, suggestions):
    # Create PDF with compression
    pdf = FPDF()
    pdf.add_page()
    
    # Enable compression
    pdf.set_compression(True)
    
    # Add header with logo and title
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(76, 175, 80)  # Medical green
    pdf.cell(200, 15, "Your Health Report Summary", ln=True, align="C")
    
    # Add subtitle and date
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(128, 128, 128)  # Gray
    pdf.cell(200, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    
    # Add quick overview section with styled header
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Quick Overview", ln=True, fill=True)
    pdf.ln(2)
    
    # Add overview content
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)  # Black
    overview = """This comprehensive report provides a detailed analysis of your blood test results. Each test is carefully evaluated and marked as Normal, High, Low, or Borderline to help you understand your health status. The report includes explanations of abnormal values, recommended actions, and important health insights."""
    pdf.multi_cell(0, 8, overview)
    pdf.ln(5)
    
    # Add test summary section
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Test Summary", ln=True, fill=True)
    pdf.ln(2)
    
    # Add summary statistics
    total_tests = len(df)
    normal_tests = len(df[df['Category'] == 'Normal'])
    abnormal_tests = len(df[df['Category'] != 'Normal'])
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)  # Black
    summary = f"""Total Tests: {total_tests}
Normal Results: {normal_tests}
Abnormal Results: {abnormal_tests}
"""
    pdf.multi_cell(0, 8, summary)
    pdf.ln(5)
    
    # Add risk summary section
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Key Findings", ln=True, fill=True)
    pdf.ln(2)
    
    # Add risk summary content
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)  # Black
    risk_summary = generate_risk_summary(df)
    risk_summary = risk_summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, risk_summary)
    pdf.ln(5)
    
    # Add detailed results section
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Your Test Results", ln=True, fill=True)
    pdf.ln(2)
    
    # Add table header
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(240, 240, 240)  # Light gray
    pdf.set_text_color(0, 0, 0)  # Black
    pdf.cell(60, 8, "Test Name", 1, 0, 'C', True)
    pdf.cell(30, 8, "Your Value", 1, 0, 'C', True)
    pdf.cell(40, 8, "Normal Range", 1, 0, 'C', True)
    pdf.cell(30, 8, "Status", 1, 0, 'C', True)
    pdf.cell(30, 8, "Unit", 1, 1, 'C', True)
    
    # Add table content
    pdf.set_font("Arial", "", 9)
    for i, row in df.iterrows():
        # Set row background color based on status
        if row['Category'] == "High":
            pdf.set_fill_color(255, 235, 235)  # Light red
        elif row['Category'] == "Low":
            pdf.set_fill_color(235, 235, 255)  # Light blue
        elif row['Category'] == "Borderline":
            pdf.set_fill_color(255, 255, 235)  # Light yellow
        else:
            pdf.set_fill_color(235, 255, 235)  # Light green
            
        # Set text color based on status
        if row['Category'] == "High":
            pdf.set_text_color(220, 53, 69)  # Red
        elif row['Category'] == "Low":
            pdf.set_text_color(0, 123, 255)  # Blue
        elif row['Category'] == "Borderline":
            pdf.set_text_color(255, 193, 7)  # Yellow
        else:
            pdf.set_text_color(40, 167, 69)  # Green
            
        # Clean and encode row data
        test = str(row['Test']).encode('latin-1', 'replace').decode('latin-1')
        value = str(row['Value']).encode('latin-1', 'replace').decode('latin-1')
        normal_range = str(row['Normal Range']).encode('latin-1', 'replace').decode('latin-1')
        category = str(row['Category']).encode('latin-1', 'replace').decode('latin-1')
        unit = str(row['Unit']).encode('latin-1', 'replace').decode('latin-1')
            
        # Add row content
        pdf.cell(60, 8, test, 1, 0, 'L', True)
        pdf.cell(30, 8, value, 1, 0, 'C', True)
        pdf.cell(40, 8, normal_range, 1, 0, 'C', True)
        pdf.cell(30, 8, category, 1, 0, 'C', True)
        pdf.cell(30, 8, unit, 1, 1, 'C', True)
    
    pdf.ln(5)
    
    # Add detailed analysis section - only for abnormal results
    abnormal_results = df[df['Category'].isin(['High', 'Low', 'Borderline'])]
    if not abnormal_results.empty:
        pdf.set_fill_color(76, 175, 80)  # Medical green background
        pdf.set_text_color(255, 255, 255)  # White text
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, "Understanding Your Results", ln=True, fill=True)
        pdf.ln(2)
        
        # Add explanations and suggestions
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(0, 0, 0)  # Black
        
        for i, row in abnormal_results.iterrows():
            # Clean and encode test name
            test_name = str(row['Test']).encode('latin-1', 'replace').decode('latin-1')
            
            # Add test name with styled header
            pdf.set_fill_color(240, 240, 240)  # Light gray background
            pdf.set_text_color(76, 175, 80)  # Medical green text
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 8, f"{test_name} ({row['Category']})", ln=True, fill=True)
            pdf.ln(2)
            
            # Add explanation
            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(0, 0, 0)  # Black
            explanation = explanations[i].encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, explanation)
            
            # Add suggestion with highlighted action items
            pdf.set_font("Arial", "B", 9)
            pdf.set_text_color(76, 175, 80)  # Medical green
            pdf.cell(0, 8, "Recommended Actions:", ln=True)
            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(0, 0, 0)  # Black
            
            # Process suggestions to highlight action items
            suggestion = suggestions[i].encode('latin-1', 'replace').decode('latin-1')
            # Split suggestions into lines
            action_items = suggestion.split('.')
            for item in action_items:
                item = item.strip()
                if item:
                    # Check if it's an action item (starts with common action words)
                    if any(item.lower().startswith(word) for word in ['avoid', 'increase', 'decrease', 'maintain', 'take', 'eat', 'drink', 'exercise', 'consult', 'schedule', 'monitor', 'reduce', 'limit', 'address', 'consider', 'follow', 'implement', 'practice', 'prevent', 'manage']):
                        # Highlight action items with a more professional style
                        pdf.set_fill_color(240, 248, 255)  # Light blue background
                        pdf.set_text_color(0, 0, 0)  # Black text
                        pdf.cell(0, 8, f"- {item}.", ln=True, fill=True)
                    else:
                        # Regular text
                        pdf.set_text_color(0, 0, 0)  # Black text
                        pdf.multi_cell(0, 8, f"- {item}.")
            pdf.ln(3)
    
    # Add health insights section
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "Health Insights", ln=True, fill=True)
    pdf.ln(2)
    
    # Add health insights content
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)  # Black
    insights = """Based on your test results, here are some key insights:
* Your overall blood cell counts are within normal ranges
* Pay special attention to the abnormal values identified
* Regular monitoring of these values is recommended
* Consider lifestyle modifications as suggested
* Schedule follow-up tests as recommended by your doctor"""
    pdf.multi_cell(0, 8, insights)
    pdf.ln(5)
    
    # Add footer with important notes
    pdf.set_y(-30)
    pdf.set_fill_color(76, 175, 80)  # Medical green background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Arial", "B", 9)
    pdf.cell(0, 8, "Important Notes:", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(0, 0, 0)  # Black
    notes = [
        "- This report is for informational purposes only.",
        "- Always consult your healthcare provider for medical advice.",
        "- Test results should be interpreted in the context of your overall health.",
        "- Regular health check-ups are recommended.",
        "- Keep a record of your test results for future reference."
    ]
    
    for note in notes:
        pdf.cell(0, 6, note.encode('latin-1', 'replace').decode('latin-1'), ln=True)
    
    # Save to temporary file with compression
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        with open(tmp.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="your_health_report.pdf" class="download-link">📄 Download Your Health Report</a>'
            return href

# Process uploaded file
if uploaded_file:
    with st.spinner("Processing your medical report..."):
        if uploaded_file.type == "application/pdf":
            pages = extract_images_from_pdf(uploaded_file)
        else:
            pages = [Image.open(uploaded_file)]

        full_text = ""
        for img in pages:
            # Create a container for the image with custom styling
            with st.container():
                st.markdown("""
                <div style='text-align: center; margin: 1rem 0;'>
                    <h4 style='color: #4CAF50; margin-bottom: 0.5rem;'>Uploaded Report Preview</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Display image with controlled size
                st.image(img, use_column_width=True, width=400)
                
                # Add a subtle border and shadow
                st.markdown("""
                <style>
                    [data-testid="stImage"] {
                        border-radius: 10px;
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                        margin: 1rem 0;
                    }
                </style>
                """, unsafe_allow_html=True)
            
            text = extract_text_from_image(img)
            if not text:
                st.error("No text could be extracted. Please ensure the image is clear and readable.")
                st.stop()
            full_text += text + "\n"

        # Parse and display results
        lab_df = parse_lab_results(full_text)
        
        if lab_df.empty:
            st.warning("""
            <div class="premium-card">
                <h3>No Test Results Found</h3>
                <p>Please ensure:</p>
                <ul>    
                    <li>The image is clear and readable</li>
                    <li>The test results are in a standard format</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display results in a clean, professional format
            st.markdown("""
            <div class="premium-card slide-up">
                <h3><div class="card-icon icon-analysis">📊</div>Test Results Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Style the dataframe
            def highlight_category(val):
                if val == "High":
                    return 'background-color: rgba(245, 101, 101, 0.2); color: #f56565; font-weight: 600;'
                elif val == "Low":
                    return 'background-color: rgba(59, 130, 246, 0.2); color: #3b82f6; font-weight: 600;'
                elif val == "Borderline":
                    return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: 600;'
                return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981; font-weight: 600;'
            
            styled_df = lab_df.style.applymap(highlight_category, subset=['Category'])
            st.dataframe(styled_df, use_container_width=True)

            # Add risk summary
            st.markdown("""
            <div class="premium-card slide-up">
                <h3><div class="card-icon icon-results">🔍</div>Risk Assessment Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            risk_summary = generate_risk_summary(lab_df)
            st.info(risk_summary)

            # Add explanations in a clean format
            st.markdown("""
            <div class="premium-card slide-up">
                <h3><div class="card-icon icon-analysis">📝</div>Detailed Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            explanations = []
            suggestions = []

            for idx, row in lab_df.iterrows():
                with st.expander(f"{row['Test']} ({row['Category']})"):
                    explanation = generate_explanation(row['Test'], row['Value'], f"{row['Normal Range']} {row['Unit']}")
                    suggestion = generate_suggestions(row['Test'], row['Value'])
                    explanations.append(explanation)
                    suggestions.append(suggestion)

                    st.markdown(f"""
                    <div class="premium-card">
                        <h4>Explanation</h4>
                        <p>{explanation}</p>
                        <h4>Recommended Actions</h4>
                        <p>{suggestion}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Add export option
            st.markdown("""
            <div class="premium-card slide-up">
                <h3><div class="card-icon icon-export">📄</div>Export Report</h3>
            </div>
            """, unsafe_allow_html=True)
            
            download_link = generate_pdf_report(lab_df, explanations, suggestions)
            st.markdown(f"""
            <div style='text-align: center; margin-top: 1rem;'>
                {download_link}
            </div>
            """, unsafe_allow_html=True)

# Update sidebar with Streamlit native components
st.sidebar.title("MediScan AI")

# How to Use section
st.sidebar.subheader("How to Use")
st.sidebar.markdown("""
1. Upload your medical report (PDF/Image)
2. Wait for AI analysis
3. Review your results
4. Download detailed report
""")

# Supported Tests section
st.sidebar.subheader("Supported Tests")
st.sidebar.markdown("""
- 🩸 Complete Blood Count
- 🩺 Blood Cell Analysis
- 📊 Blood Indices
- 🔬 Differential Count
""")

# Need Help section
st.sidebar.subheader("Need Help?")
st.sidebar.markdown("For support or questions:")
st.sidebar.markdown("""
- 📧 support@mediscan.ai
- 📞 1-800-MEDISCAN
""")

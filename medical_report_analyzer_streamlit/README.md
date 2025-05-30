# Ai Medical Report Analyzer

An AI-powered web application that analyzes medical reports using OCR and Generative AI to provide detailed explanations, risk assessments, and lifestyle suggestions.

## Features

- **Document Processing**
  - Upload medical reports in PDF or image formats
  - OCR (Optical Character Recognition) for text extraction
  - Support for both scanned documents and digital files

- **AI-Powered Analysis**
  - Detailed explanations of medical test results
  - Risk assessment and categorization
  - Personalized lifestyle suggestions
  - Normal range comparisons

- **User-Friendly Interface**
  - Modern, responsive design
  - Interactive data visualization
  - Real-time processing feedback
  - Dark mode interface

- **Export Capabilities**
  - Generate comprehensive PDF reports
  - Download analysis results
  - Share findings with healthcare providers

## Project Structure

```
medical_report_analyzer_streamlit/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies
├── create_test_image.py  # Test image generation
├── sample_report.txt     # Sample medical report
├── test_app.py          # Unit tests
└── .streamlit/          # Streamlit configuration
```

## Dependencies

- **Core Dependencies**
  - streamlit==1.31.1
  - opencv-python==4.8.1.78
  - pytesseract==0.3.10
  - google-generativeai==0.3.2
  - python-dotenv==1.0.0

- **PDF Processing**
  - pypdf2==3.0.1
  - pdf2image==1.16.3
  - reportlab==4.0.9

- **Image Processing**
  - numpy==1.26.2
  - pillow==10.1.0

## Acknowledgments

- Google Gemini Pro for AI capabilities
- Tesseract OCR for text extraction
- Streamlit for the web interface framework
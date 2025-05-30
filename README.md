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

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine
- Google Cloud API key (for Gemini Pro)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medical_report_analyzer_streamlit
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

5. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Google Cloud API key: `GOOGLE_API_KEY=your_api_key_here`

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your medical report (PDF or image format)

4. Wait for the analysis to complete

5. Review the results, explanations, and suggestions

6. Download the PDF report if needed

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini Pro for AI capabilities
- Tesseract OCR for text extraction
- Streamlit for the web interface framework
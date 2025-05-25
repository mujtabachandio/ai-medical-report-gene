# Medical Report Analyzer

An advanced AI-powered application that analyzes medical lab reports, extracts test results, and provides intelligent insights using computer vision and natural language processing.

## Features

- 📄 **Multi-format Support**: Upload PDF or image files of medical reports
- 🔍 **Intelligent Analysis**: Extracts and analyzes lab test results
- 📊 **Visual Analytics**: Interactive charts and graphs of test results
- ⚠️ **Risk Assessment**: Identifies critical and abnormal values
- 📈 **Trend Analysis**: Tracks changes in test values over time
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔒 **Privacy-Focused**: Processes data locally, no data storage

## Prerequisites

- Python 3.9 or higher
- Tesseract OCR engine
- Google API key (for advanced analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-report-analyzer.git
cd medical-report-analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **Windows**: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **MacOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

5. Set up environment variables:
Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your medical report (PDF or image)

4. View the analysis results and insights

## Project Structure

```
medical_report_analyzer_streamlit/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── streamlit.toml        # Streamlit configuration
├── .env                  # Environment variables
└── README.md            # Project documentation
```

## Key Technologies

- **Streamlit**: Web application framework
- **OpenCV**: Image processing
- **Tesseract**: OCR (Optical Character Recognition)
- **Google Generative AI**: Advanced text analysis
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations

## Supported Test Types

- Complete Blood Count (CBC)
- Lipid Profile
- Liver Function Tests
- Kidney Function Tests
- Thyroid Function Tests
- And more...

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Streamlit team for the amazing framework
- Google for the Generative AI API
- Tesseract OCR community

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Roadmap

- [ ] Add support for more test types
- [ ] Implement machine learning for better analysis
- [ ] Add multi-language support
- [ ] Create mobile app version
- [ ] Add batch processing capability

## Security

- All processing is done locally
- No data is stored on servers
- Secure file handling
- Input validation and sanitization

## Performance

- Optimized image processing
- Cached results for faster analysis
- Efficient memory management
- Responsive UI design 
# SmartMRI Planner - README

## Overview

SmartMRI Planner is an AI-powered application that generates personalized MRI protocol recommendations based on patient information and research literature. By analyzing research papers and patient clinical data, the application suggests optimal MRI protocols, sequences, field strengths, and special considerations tailored to each patient's specific conditions.

## Key Features

- **Research Paper Analysis**: Upload PDF research articles or provide URLs to academic papers
- **Patient Data Processing**: Enter patient clinical information in natural language format
- **AI-Powered Recommendations**: Receive personalized MRI protocol recommendations using LLM/NLP technology
- **Comprehensive Output**: View recommended sequences, field strength, special considerations, rationale, and alternative options

## Architecture

SmartMRI Planner consists of four main components:

1. **PDF Processor**: Extracts text from PDF research papers and URLs
2. **NLP Analyzer**: Processes patient information and research findings using LangChain and OpenAI
3. **Protocol Recommender**: Generates personalized MRI protocol recommendations
4. **Web Interface**: Flask-based user interface for interacting with the system

## Installation

### Prerequisites

- Python 3.10 or higher
- poppler-utils (for PDF processing)
- OpenAI API key

### Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SmartMRI_Planner.git
   cd SmartMRI_Planner
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   SECRET_KEY=your-secret-key-here
   ```

4. Run the application:
   ```
   python src/app.py
   ```

5. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Enter patient information in the provided text area
2. Upload PDF research papers or provide URLs to academic articles
3. Click "Generate Protocol Recommendation"
4. Review the personalized MRI protocol recommendation

## Example

**Patient Information:**
```
Patient is a 58 year old male with a history of stage 2 hypertension and eGFR of 45mL/min/1.73m2. Assess for fibrosis.
```

**Recommendation Output:**
- **Sequences**: T1 mapping, T2 mapping, Native T1
- **Field Strength**: 3T
- **Special Considerations**: Breath-held acquisitions, Non-contrast protocol
- **Rationale**: Based on the patient's stage 2 hypertension and reduced kidney function (eGFR 45), a non-contrast protocol using native T1 and T2 mapping at 3T with breath-held acquisitions is recommended for optimal assessment of fibrosis while minimizing risks.

## Documentation

For more detailed information, please refer to:

- [User Guide](docs/user_guide.md): Comprehensive guide for users
- [Technical Documentation](docs/technical_documentation.md): Detailed technical information for developers
- [Deployment Guide](docs/deployment_guide.md): Instructions for deploying the application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the API for NLP processing
- LangChain for the LLM application framework
- Flask team for the web framework

## Disclaimer

SmartMRI Planner is designed as a decision support tool and should not replace professional medical judgment. Always consult with qualified radiologists and healthcare providers before finalizing MRI protocols for patients.
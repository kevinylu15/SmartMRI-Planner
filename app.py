"""
Flask application for SmartMRI Planner

This module serves as the web interface for the SmartMRI Planner application,
connecting the UI with the integrated SmartMRI Planner backend.
"""

import os
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

# Import our modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.integration import SmartMRIPlanner

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-key-for-testing')
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# This will be initialized when API key is provided
planner = None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    """
    Process the uploaded files, URLs, and patient information.
    
    Returns:
        JSON response with protocol recommendation
    """
    try:
        # Check if OpenAI API key is provided
        api_key = request.form.get('api_key')
        if not api_key and not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OpenAI API key is required. Please provide it in the form or set the OPENAI_API_KEY environment variable.'
            }), 400
        
        # Initialize SmartMRI Planner if not already done
        global planner
        if not planner:
            planner = SmartMRIPlanner(api_key)
        
        # Get patient information
        patient_text = request.form.get('patient_info', '')
        if not patient_text:
            return jsonify({'error': 'Patient information is required'}), 400
        
        # Get paper URLs
        paper_urls = request.form.get('paper_urls', '').strip().split('\n')
        paper_urls = [url.strip() for url in paper_urls if url.strip()]
        
        # Process uploaded files
        uploaded_files = []
        if 'papers' in request.files:
            files = request.files.getlist('papers')
            for file in files:
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    uploaded_files.append(filepath)
        
        # Combine all paper sources
        paper_sources = paper_urls + uploaded_files
        
        # Check if we have any papers to analyze
        if not paper_sources:
            return jsonify({
                'error': 'No valid papers provided. Please upload PDF files or provide valid URLs.'
            }), 400
        
        # Process the complete workflow
        result = planner.process_complete_workflow(patient_text, paper_sources)
        
        # Format response
        recommendation = result['recommendation']
        metadata = result['metadata']
        
        response = {
            'sequences': recommendation['sequences'],
            'field_strength': recommendation['field_strength'],
            'contrast_agent': recommendation['contrast_agent'],
            'special_considerations': recommendation['special_considerations'],
            'rationale': recommendation['rationale'],
            'alternative_options': recommendation['alternative_options'],
            'contraindications': recommendation['contraindications'],
            'analyzed_papers': metadata['processed_sources']
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temporary files
        for filepath in uploaded_files:
            try:
                os.remove(filepath)
            except:
                pass
        
        # Clean up planner resources
        if planner:
            planner.cleanup()

@app.route('/api/test', methods=['GET'])
def test():
    """
    Test endpoint that returns a mock recommendation.
    Used for testing the UI without making actual API calls.
    
    Returns:
        JSON response with mock protocol recommendation
    """
    # Mock recommendation data - this simulates the output from the integration module
    result = {
        'recommendation': {
            'sequences': ["T1 mapping", "T2 mapping", "Native T1"],
            'field_strength': "3T",
            'contrast_agent': "None (non-contrast protocol)",
            'special_considerations': [
                "Breath-held acquisitions to improve image quality",
                "Non-contrast protocol due to reduced kidney function (eGFR 45)"
            ],
            'rationale': "Based on the patient's stage 2 hypertension and reduced kidney function (eGFR 45), a non-contrast protocol using native T1 and T2 mapping at 3T with breath-held acquisitions is recommended for optimal assessment of fibrosis while minimizing risks.",
            'alternative_options': [
                {
                    'sequences': ["T1 mapping", "T2 mapping"],
                    'field_strength': "1.5T",
                    'rationale': "If 3T is not available, 1.5T can be used with slightly reduced sensitivity."
                }
            ],
            'contraindications': [
                "Gadolinium-based contrast agents are relatively contraindicated due to reduced kidney function."
            ]
        },
        'metadata': {
            'processed_sources': [
                "Smith et al. (2024) - Advanced MRI Protocols for Cardiac Fibrosis",
                "Johnson et al. (2023) - MRI Assessment in Patients with Reduced Kidney Function"
            ],
            'patient_info': {
                'age': 58,
                'gender': 'male',
                'conditions': [{'entity_type': 'condition', 'name': 'stage 2 hypertension'}],
                'measurements': [{'entity_type': 'measurement', 'name': 'eGFR', 'value': '45mL/min/1.73m2'}],
                'assessment_goal': 'Assess for fibrosis'
            }
        }
    }
    
    # Format response to match the structure expected by the frontend
    recommendation = result['recommendation']
    metadata = result['metadata']
    
    response = {
        'sequences': recommendation['sequences'],
        'field_strength': recommendation['field_strength'],
        'contrast_agent': recommendation['contrast_agent'],
        'special_considerations': recommendation['special_considerations'],
        'rationale': recommendation['rationale'],
        'alternative_options': recommendation['alternative_options'],
        'contraindications': recommendation['contraindications'],
        'analyzed_papers': metadata['processed_sources']
    }
    
    return jsonify(response)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum file size is 16 MB.'}), 413

@app.errorhandler(500)
def server_error(error):
    """Handle server errors."""
    return jsonify({'error': 'An internal server error occurred. Please try again later.'}), 500

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)

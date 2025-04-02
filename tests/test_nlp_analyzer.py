"""
Test script for NLP Analyzer module

This script tests the functionality of the NLP analyzer module with sample data.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from nlp_analyzer import NLPAnalyzer, PatientInfo, ResearchFindings

def test_nlp_analyzer():
    """Test the NLP analyzer with sample data."""
    print("Testing NLP Analyzer module...")
    
    # Set up a mock API key for testing
    # In a real environment, this would be set in a .env file or environment variable
    os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
    
    try:
        # Initialize the NLP analyzer
        analyzer = NLPAnalyzer()
        
        # Test patient info extraction
        print("\nTesting patient info extraction...")
        patient_text = "Patient is a 58 year old male with a history of stage 2 hypertension and eGFR of 45mL/min/1.73m2. Assess for fibrosis."
        
        # Mock the LLM response for testing without actual API calls
        mock_patient_info = PatientInfo(
            age=58,
            gender="male",
            conditions=[{"entity_type": "condition", "name": "stage 2 hypertension", "value": None, "context": "history of"}],
            measurements=[{"entity_type": "measurement", "name": "eGFR", "value": "45mL/min/1.73m2", "context": None}],
            assessment_goal="Assess for fibrosis"
        )
        
        print(f"Patient text: {patient_text}")
        print(f"Extracted patient info (mocked): {json.dumps(mock_patient_info.dict(), indent=2)}")
        
        # Test research paper analysis
        print("\nTesting research paper analysis...")
        paper_text = """
        Abstract
        This study evaluates the effectiveness of various MRI protocols for detecting cardiac fibrosis in patients with hypertension and reduced kidney function. We found that T1 mapping at 3T provided the highest sensitivity for detecting diffuse fibrosis.
        
        Methods
        We used T1 and T2 mapping sequences at both 1.5T and 3T field strengths. For patients with reduced kidney function (eGFR < 60mL/min/1.73m2), non-contrast protocols were preferred.
        
        Results
        Our analysis found that breath-held acquisitions improved image quality significantly. For patients with stage 2 hypertension, a combination of native T1 and T2 mapping at 3T provided optimal diagnostic value.
        """
        
        # Mock the LLM response for testing without actual API calls
        mock_findings = ResearchFindings(
            mri_protocols=[
                {"name": "Non-contrast protocol", "indication": "Reduced kidney function"},
                {"name": "Native T1 and T2 mapping", "indication": "Stage 2 hypertension"}
            ],
            field_strengths=["1.5T", "3T"],
            sequences=["T1 mapping", "T2 mapping", "Native T1"],
            conditions=["Hypertension", "Reduced kidney function", "Cardiac fibrosis"],
            special_considerations=[
                {"consideration": "Breath-held acquisitions", "benefit": "Improved image quality"},
                {"consideration": "Non-contrast protocol", "benefit": "Safer for reduced kidney function"}
            ],
            key_findings=[
                "T1 mapping at 3T provided highest sensitivity for detecting diffuse fibrosis",
                "Breath-held acquisitions improved image quality",
                "Native T1 and T2 mapping at 3T optimal for stage 2 hypertension"
            ]
        )
        
        print(f"Paper text sample: {paper_text[:200]}...")
        print(f"Extracted research findings (mocked): {json.dumps(mock_findings.dict(), indent=2)}")
        
        print("\nNLP Analyzer testing completed with mock data.")
        print("Note: In a production environment, actual API calls would be made to OpenAI.")
        
    except Exception as e:
        print(f"Error testing NLP analyzer: {e}")

if __name__ == "__main__":
    test_nlp_analyzer()

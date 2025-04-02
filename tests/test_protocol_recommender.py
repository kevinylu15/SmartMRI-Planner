"""
Test script for Protocol Recommender module

This script tests the functionality of the protocol recommender module with sample data.
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from protocol_recommender import ProtocolRecommender, ProtocolRecommendation
from nlp_analyzer import PatientInfo, ResearchFindings, MedicalEntity

def test_protocol_recommender():
    """Test the protocol recommender with sample data."""
    print("Testing Protocol Recommender module...")
    
    # Set up a mock API key for testing
    # In a real environment, this would be set in a .env file or environment variable
    os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
    
    try:
        # Create mock patient info
        patient_info = PatientInfo(
            age=58,
            gender="male",
            conditions=[MedicalEntity(entity_type="condition", name="stage 2 hypertension")],
            measurements=[MedicalEntity(entity_type="measurement", name="eGFR", value="45mL/min/1.73m2")],
            assessment_goal="Assess for fibrosis"
        )
        
        # Create mock research findings
        research_findings = ResearchFindings(
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
        
        print("\nPatient Information:")
        print(json.dumps(patient_info.model_dump(), indent=2))
        
        print("\nResearch Findings:")
        print(json.dumps(research_findings.model_dump(), indent=2))
        
        # Initialize the protocol recommender
        recommender = ProtocolRecommender()
        
        # Test kidney function check
        kidney_function = recommender._check_kidney_function(patient_info)
        print("\nKidney Function Check:")
        print(json.dumps(kidney_function, indent=2))
        
        # Test conditions check
        conditions = recommender._check_conditions(patient_info)
        print("\nConditions Check:")
        print(json.dumps(conditions, indent=2))
        
        # Mock the recommendation generation (without actual API call)
        mock_recommendation = ProtocolRecommendation(
            sequences=["T1 mapping", "T2 mapping", "Native T1"],
            field_strength="3T",
            contrast_agent=None,
            special_considerations=[
                "Breath-held acquisitions to improve image quality",
                "Non-contrast protocol due to reduced kidney function (eGFR 45)"
            ],
            rationale="Based on the patient's stage 2 hypertension and reduced kidney function (eGFR 45), " +
                     "a non-contrast protocol using native T1 and T2 mapping at 3T with breath-held acquisitions " +
                     "is recommended for optimal assessment of fibrosis while minimizing risks.",
            alternative_options=[
                {
                    "sequences": ["T1 mapping", "T2 mapping"],
                    "field_strength": "1.5T",
                    "rationale": "If 3T is not available, 1.5T can be used with slightly reduced sensitivity."
                }
            ],
            contraindications=[
                "Gadolinium-based contrast agents are relatively contraindicated due to reduced kidney function."
            ]
        )
        
        print("\nMock Protocol Recommendation:")
        print(json.dumps(mock_recommendation.model_dump(), indent=2))
        
        # Test direct text recommendation (mock)
        patient_text = "Patient is a 58 year old male with a history of stage 2 hypertension and eGFR of 45mL/min/1.73m2. Assess for fibrosis."
        research_text = [
            "This study evaluates the effectiveness of various MRI protocols for detecting cardiac fibrosis in patients with hypertension and reduced kidney function. We found that T1 mapping at 3T provided the highest sensitivity for detecting diffuse fibrosis."
        ]
        
        print("\nDirect Text Recommendation Input:")
        print(f"Patient Text: {patient_text}")
        print(f"Research Text: {research_text[0]}")
        
        print("\nNote: In a production environment, actual API calls would be made to OpenAI.")
        print("Protocol Recommender testing completed with mock data.")
        
    except Exception as e:
        print(f"Error testing protocol recommender: {e}")

if __name__ == "__main__":
    test_protocol_recommender()

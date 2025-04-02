"""
Protocol Recommendation Engine for SmartMRI Planner

This module generates personalized MRI protocol recommendations based on
patient information and research findings.
"""

import os
import json
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

# Import models from NLP analyzer
from nlp_analyzer import PatientInfo, ResearchFindings, MedicalEntity
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


class ProtocolRecommendation(BaseModel):
    """Model for MRI protocol recommendations."""
    sequences: List[str] = Field(default_factory=list, description="Recommended MRI sequences")
    field_strength: str = Field(description="Recommended field strength (e.g., 1.5T, 3T)")
    contrast_agent: Optional[str] = Field(None, description="Recommended contrast agent, if any")
    special_considerations: List[str] = Field(default_factory=list, description="Special considerations for the protocol")
    rationale: str = Field(description="Rationale for the recommendation")
    alternative_options: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative protocol options")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications to consider")


class ProtocolRecommender:
    """Class for generating MRI protocol recommendations."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the protocol recommender.
        
        Args:
            openai_api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        # Use provided API key or get from environment
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide it directly.")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.2,  # Slightly higher temperature for more diverse recommendations
            api_key=self.api_key
        )
    
    def _check_kidney_function(self, patient_info: PatientInfo) -> Dict[str, Any]:
        """
        Check kidney function to determine contrast agent suitability.
        
        Args:
            patient_info: Structured patient information
            
        Returns:
            Dictionary with kidney function assessment
        """
        kidney_function = {
            "reduced_function": False,
            "egfr_value": None,
            "contrast_contraindicated": False
        }
        
        # Check for eGFR measurements
        for measurement in patient_info.measurements:
            if "egfr" in measurement.name.lower() and measurement.value:
                try:
                    # Extract numeric value from eGFR string
                    import re
                    egfr_match = re.search(r'(\d+(\.\d+)?)', measurement.value)
                    if egfr_match:
                        egfr_value = float(egfr_match.group(1))
                        kidney_function["egfr_value"] = egfr_value
                        
                        # Check if eGFR indicates reduced kidney function
                        if egfr_value < 60:
                            kidney_function["reduced_function"] = True
                        
                        # Check if eGFR contraindicates contrast
                        if egfr_value < 30:
                            kidney_function["contrast_contraindicated"] = True
                except Exception as e:
                    print(f"Error parsing eGFR value: {e}")
        
        return kidney_function
    
    def _check_conditions(self, patient_info: PatientInfo) -> Dict[str, bool]:
        """
        Check for specific conditions that may affect protocol selection.
        
        Args:
            patient_info: Structured patient information
            
        Returns:
            Dictionary with condition flags
        """
        conditions = {
            "hypertension": False,
            "diabetes": False,
            "cardiac_disease": False,
            "fibrosis_assessment": False
        }
        
        # Check for specific conditions
        for condition in patient_info.conditions:
            condition_name = condition.name.lower()
            
            if "hypertension" in condition_name:
                conditions["hypertension"] = True
            
            if "diabetes" in condition_name:
                conditions["diabetes"] = True
            
            if any(term in condition_name for term in ["cardiac", "heart", "coronary"]):
                conditions["cardiac_disease"] = True
        
        # Check assessment goal for fibrosis
        if patient_info.assessment_goal and "fibrosis" in patient_info.assessment_goal.lower():
            conditions["fibrosis_assessment"] = True
        
        return conditions
    
    def generate_recommendation(self, patient_info: PatientInfo, research_findings: ResearchFindings) -> ProtocolRecommendation:
        """
        Generate MRI protocol recommendation based on patient info and research findings.
        
        Args:
            patient_info: Structured patient information
            research_findings: Structured research findings
            
        Returns:
            Protocol recommendation
        """
        # Check kidney function and conditions
        kidney_function = self._check_kidney_function(patient_info)
        conditions = self._check_conditions(patient_info)
        
        # Create a context dictionary for recommendation generation
        context = {
            "patient_age": patient_info.age,
            "patient_gender": patient_info.gender,
            "kidney_function": kidney_function,
            "conditions": conditions,
            "assessment_goal": patient_info.assessment_goal,
            "available_sequences": research_findings.sequences,
            "available_field_strengths": research_findings.field_strengths,
            "research_key_findings": research_findings.key_findings,
            "special_considerations_from_research": research_findings.special_considerations
        }
        
        # Use LLM to generate recommendation
        parser = PydanticOutputParser(pydantic_object=ProtocolRecommendation)
        
        prompt_template = PromptTemplate(
            template="""
            You are an expert radiologist specializing in MRI protocol selection.
            
            Generate a personalized MRI protocol recommendation based on the following patient information and research findings.
            
            Patient Information:
            - Age: {patient_age}
            - Gender: {patient_gender}
            - Kidney Function: eGFR {kidney_function_details}
            - Conditions: {conditions_details}
            - Assessment Goal: {assessment_goal}
            
            Research Findings:
            - Available Sequences: {available_sequences}
            - Available Field Strengths: {available_field_strengths}
            - Key Research Findings: {research_key_findings}
            - Special Considerations: {special_considerations}
            
            Based on this information, provide a detailed MRI protocol recommendation.
            
            {format_instructions}
            """,
            input_variables=[
                "patient_age", "patient_gender", "kidney_function_details", 
                "conditions_details", "assessment_goal", "available_sequences",
                "available_field_strengths", "research_key_findings", "special_considerations"
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Format context for prompt
        formatted_context = {
            "patient_age": context["patient_age"] or "Unknown",
            "patient_gender": context["patient_gender"] or "Unknown",
            "kidney_function_details": f"Value: {kidney_function['egfr_value'] or 'Unknown'}, " +
                                      f"Reduced Function: {kidney_function['reduced_function']}, " +
                                      f"Contrast Contraindicated: {kidney_function['contrast_contraindicated']}",
            "conditions_details": ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in conditions.items()]),
            "assessment_goal": context["assessment_goal"] or "Not specified",
            "available_sequences": ", ".join(context["available_sequences"]) if context["available_sequences"] else "Standard sequences",
            "available_field_strengths": ", ".join(context["available_field_strengths"]) if context["available_field_strengths"] else "1.5T, 3T",
            "research_key_findings": "\n- " + "\n- ".join(context["research_key_findings"]) if context["research_key_findings"] else "No specific findings",
            "special_considerations": "\n- " + "\n- ".join([f"{sc.get('consideration', '')}: {sc.get('benefit', '')}" 
                                                         for sc in context["special_considerations_from_research"]]) 
                                    if context["special_considerations_from_research"] else "No special considerations"
        }
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.run(**formatted_context)
        
        # Parse result
        try:
            return parser.parse(result)
        except Exception as e:
            print(f"Error parsing recommendation: {e}")
            # Return default recommendation if parsing fails
            return ProtocolRecommendation(
                sequences=["T1-weighted", "T2-weighted"],
                field_strength="1.5T",
                special_considerations=["Standard protocol due to parsing error"],
                rationale="Error occurred during recommendation generation. Using standard protocol as fallback."
            )
    
    def generate_recommendation_from_text(self, patient_text: str, research_texts: List[str]) -> ProtocolRecommendation:
        """
        Generate MRI protocol recommendation directly from text inputs.
        
        Args:
            patient_text: Text containing patient information
            research_texts: List of texts from research papers
            
        Returns:
            Protocol recommendation
        """
        # This is a simplified version that would use the NLP analyzer in a real implementation
        # For demonstration purposes, we'll use a mock implementation
        
        parser = PydanticOutputParser(pydantic_object=ProtocolRecommendation)
        
        prompt_template = PromptTemplate(
            template="""
            You are an expert radiologist specializing in MRI protocol selection.
            
            Generate a personalized MRI protocol recommendation based on the following patient information and research findings.
            
            Patient Information:
            {patient_text}
            
            Research Findings:
            {research_text}
            
            Based on this information, provide a detailed MRI protocol recommendation.
            
            {format_instructions}
            """,
            input_variables=["patient_text", "research_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.run(patient_text=patient_text, research_text="\n\n".join(research_texts))
        
        # Parse result
        try:
            return parser.parse(result)
        except Exception as e:
            print(f"Error parsing recommendation: {e}")
            # Return default recommendation if parsing fails
            return ProtocolRecommendation(
                sequences=["T1-weighted", "T2-weighted"],
                field_strength="1.5T",
                special_considerations=["Standard protocol due to parsing error"],
                rationale="Error occurred during recommendation generation. Using standard protocol as fallback."
            )


# Example usage
if __name__ == "__main__":
    # Set up a mock API key for testing
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
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
    
    # Initialize recommender
    recommender = ProtocolRecommender()
    
    # Generate recommendation
    # In a real implementation, this would make an actual API call
    # For demonstration, we'll just print what would be generated
    print("In a real implementation with a valid API key, the recommendation would be generated based on:")
    print(f"Patient: {patient_info.model_dump_json(indent=2)}")
    print(f"Research: {research_findings.model_dump_json(indent=2)}")
    
    # Example of what a recommendation might look like
    example_recommendation = ProtocolRecommendation(
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
    
    print("\nExample recommendation:")
    print(example_recommendation.model_dump_json(indent=2))

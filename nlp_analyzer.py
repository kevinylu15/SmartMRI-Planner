"""
NLP Analysis Module for SmartMRI Planner

This module handles the NLP analysis of extracted text from research papers
and patient data to identify relevant medical entities and concepts.
It uses LangChain with OpenAI integration for robust medical text processing.
"""

import os
import re
import json
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

class MedicalEntity(BaseModel):
    """Model for medical entities extracted from text."""
    entity_type: str = Field(description="Type of medical entity (e.g., condition, medication, procedure)")
    name: str = Field(description="Name of the medical entity")
    value: Optional[str] = Field(None, description="Value associated with the entity (e.g., dosage, measurement)")
    context: Optional[str] = Field(None, description="Contextual information about the entity")

class PatientInfo(BaseModel):
    """Model for structured patient information."""
    age: Optional[int] = Field(None, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender")
    conditions: List[MedicalEntity] = Field(default_factory=list, description="Medical conditions")
    measurements: List[MedicalEntity] = Field(default_factory=list, description="Medical measurements (e.g., eGFR, blood pressure)")
    medications: List[MedicalEntity] = Field(default_factory=list, description="Medications")
    procedures: List[MedicalEntity] = Field(default_factory=list, description="Medical procedures")
    assessment_goal: Optional[str] = Field(None, description="The goal of the MRI assessment (e.g., 'Assess for fibrosis')")

class ResearchFindings(BaseModel):
    """Model for structured research findings from papers."""
    mri_protocols: List[Dict[str, Any]] = Field(default_factory=list, description="MRI protocols mentioned in research")
    field_strengths: List[str] = Field(default_factory=list, description="MRI field strengths mentioned in research")
    sequences: List[str] = Field(default_factory=list, description="MRI sequences mentioned in research")
    conditions: List[str] = Field(default_factory=list, description="Medical conditions discussed in research")
    special_considerations: List[Dict[str, Any]] = Field(default_factory=list, description="Special considerations for MRI protocols")
    key_findings: List[str] = Field(default_factory=list, description="Key findings from the research")

class NLPAnalyzer:
    """Class for NLP analysis of medical text using LangChain and OpenAI."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the NLP analyzer.
        
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
            temperature=0,
            api_key=self.api_key
        )
        
        # Initialize text splitter for handling large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_patient_info(self, patient_text: str) -> PatientInfo:
        """
        Extract structured patient information from text.
        
        Args:
            patient_text: Text containing patient information
            
        Returns:
            Structured patient information
        """
        # Define output parser
        parser = PydanticOutputParser(pydantic_object=PatientInfo)
        
        # Define prompt template
        prompt_template = PromptTemplate(
            template="""
            You are a medical NLP expert specializing in extracting structured information from patient data.
            
            Extract all relevant patient information from the following text and format it according to the specified JSON schema.
            Focus on age, gender, medical conditions, measurements (like eGFR, blood pressure), medications, procedures, and assessment goals.
            
            Patient text:
            {patient_text}
            
            {format_instructions}
            """,
            input_variables=["patient_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.run(patient_text=patient_text)
        
        # Parse result
        try:
            return parser.parse(result)
        except Exception as e:
            print(f"Error parsing patient info: {e}")
            # Return empty structure if parsing fails
            return PatientInfo()
    
    def analyze_research_paper(self, paper_text: str) -> ResearchFindings:
        """
        Analyze research paper text to extract relevant MRI protocol information.
        
        Args:
            paper_text: Text from a research paper
            
        Returns:
            Structured research findings
        """
        # Handle large papers by splitting into chunks
        if len(paper_text) > 4000:
            chunks = self.text_splitter.split_text(paper_text)
            all_findings = []
            
            for i, chunk in enumerate(chunks):
                # Define output parser for each chunk
                parser = PydanticOutputParser(pydantic_object=ResearchFindings)
                
                # Define prompt template
                prompt_template = PromptTemplate(
                    template="""
                    You are a medical research analyst specializing in MRI protocols.
                    
                    Extract relevant MRI protocol information from the following research paper text (chunk {chunk_num} of {total_chunks}).
                    Focus on MRI protocols, field strengths, sequences, medical conditions, special considerations, and key findings.
                    
                    Research paper text:
                    {paper_text}
                    
                    {format_instructions}
                    """,
                    input_variables=["paper_text", "chunk_num", "total_chunks"],
                    partial_variables={"format_instructions": parser.get_format_instructions()}
                )
                
                # Create and run chain
                chain = LLMChain(llm=self.llm, prompt=prompt_template)
                result = chain.run(paper_text=chunk, chunk_num=i+1, total_chunks=len(chunks))
                
                # Parse result
                try:
                    findings = parser.parse(result)
                    all_findings.append(findings)
                except Exception as e:
                    print(f"Error parsing research findings from chunk {i+1}: {e}")
            
            # Merge findings from all chunks
            merged_findings = ResearchFindings()
            for findings in all_findings:
                merged_findings.mri_protocols.extend(findings.mri_protocols)
                merged_findings.field_strengths.extend(findings.field_strengths)
                merged_findings.sequences.extend(findings.sequences)
                merged_findings.conditions.extend(findings.conditions)
                merged_findings.special_considerations.extend(findings.special_considerations)
                merged_findings.key_findings.extend(findings.key_findings)
            
            # Remove duplicates
            merged_findings.field_strengths = list(set(merged_findings.field_strengths))
            merged_findings.sequences = list(set(merged_findings.sequences))
            merged_findings.conditions = list(set(merged_findings.conditions))
            merged_findings.key_findings = list(set(merged_findings.key_findings))
            
            return merged_findings
        else:
            # For smaller papers, process in one go
            parser = PydanticOutputParser(pydantic_object=ResearchFindings)
            
            prompt_template = PromptTemplate(
                template="""
                You are a medical research analyst specializing in MRI protocols.
                
                Extract relevant MRI protocol information from the following research paper text.
                Focus on MRI protocols, field strengths, sequences, medical conditions, special considerations, and key findings.
                
                Research paper text:
                {paper_text}
                
                {format_instructions}
                """,
                input_variables=["paper_text"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            result = chain.run(paper_text=paper_text)
            
            try:
                return parser.parse(result)
            except Exception as e:
                print(f"Error parsing research findings: {e}")
                return ResearchFindings()
    
    def analyze_multiple_papers(self, paper_texts: List[str]) -> ResearchFindings:
        """
        Analyze multiple research papers and combine the findings.
        
        Args:
            paper_texts: List of texts from research papers
            
        Returns:
            Combined structured research findings
        """
        all_findings = []
        
        for paper_text in paper_texts:
            findings = self.analyze_research_paper(paper_text)
            all_findings.append(findings)
        
        # Merge findings from all papers
        merged_findings = ResearchFindings()
        for findings in all_findings:
            merged_findings.mri_protocols.extend(findings.mri_protocols)
            merged_findings.field_strengths.extend(findings.field_strengths)
            merged_findings.sequences.extend(findings.sequences)
            merged_findings.conditions.extend(findings.conditions)
            merged_findings.special_considerations.extend(findings.special_considerations)
            merged_findings.key_findings.extend(findings.key_findings)
        
        # Remove duplicates
        merged_findings.field_strengths = list(set(merged_findings.field_strengths))
        merged_findings.sequences = list(set(merged_findings.sequences))
        merged_findings.conditions = list(set(merged_findings.conditions))
        merged_findings.key_findings = list(set(merged_findings.key_findings))
        
        return merged_findings
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text.
        
        Args:
            text: Medical text
            
        Returns:
            List of extracted medical entities
        """
        parser = PydanticOutputParser(pydantic_object=List[MedicalEntity])
        
        prompt_template = PromptTemplate(
            template="""
            You are a medical NLP expert specializing in extracting medical entities from text.
            
            Extract all medical entities from the following text and format them according to the specified JSON schema.
            Focus on conditions, medications, procedures, and measurements.
            
            Text:
            {text}
            
            {format_instructions}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        result = chain.run(text=text)
        
        try:
            return parser.parse(result)
        except Exception as e:
            print(f"Error parsing medical entities: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Set up a mock API key for testing
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    analyzer = NLPAnalyzer()
    
    # Example patient text
    patient_text = "Patient is a 58 year old male with a history of stage 2 hypertension and eGFR of 45mL/min/1.73m2. Assess for fibrosis."
    
    # Extract patient info
    patient_info = analyzer.extract_patient_info(patient_text)
    print("Patient Info:", patient_info.json(indent=2))
    
    # Example research paper text (abbreviated)
    paper_text = """
    Abstract
    This study evaluates the effectiveness of various MRI protocols for detecting cardiac fibrosis in patients with hypertension and reduced kidney function. We found that T1 mapping at 3T provided the highest sensitivity for detecting diffuse fibrosis.
    
    Methods
    We used T1 and T2 mapping sequences at both 1.5T and 3T field strengths. For patients with reduced kidney function (eGFR < 60mL/min/1.73m2), non-contrast protocols were preferred.
    
    Results
    Our analysis found that breath-held acquisitions improved image quality significantly. For patients with stage 2 hypertension, a combination of native T1 and T2 mapping at 3T provided optimal diagnostic value.
    """
    
    # Analyze research paper
    findings = analyzer.analyze_research_paper(paper_text)
    print("Research Findings:", findings.json(indent=2))

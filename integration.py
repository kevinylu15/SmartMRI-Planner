"""
Integration module for SmartMRI Planner

This module provides the integration layer between the different components
of the SmartMRI Planner application: PDF processor, NLP analyzer, and protocol recommender.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple

from pdf_processor import PDFProcessor
from nlp_analyzer import NLPAnalyzer, PatientInfo, ResearchFindings
from protocol_recommender import ProtocolRecommender, ProtocolRecommendation


class SmartMRIPlanner:
    """
    Main integration class for SmartMRI Planner application.
    
    This class coordinates the workflow between the PDF processor,
    NLP analyzer, and protocol recommender components.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the SmartMRI Planner.
        
        Args:
            openai_api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide it directly.")
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.nlp_analyzer = NLPAnalyzer(self.api_key)
        self.protocol_recommender = ProtocolRecommender(self.api_key)
        
        # Create temporary directory for file processing
        self.temp_dir = tempfile.mkdtemp()
    
    def process_patient_data(self, patient_text: str) -> PatientInfo:
        """
        Process patient data to extract structured information.
        
        Args:
            patient_text: Text containing patient information
            
        Returns:
            Structured patient information
        """
        return self.nlp_analyzer.extract_patient_info(patient_text)
    
    def process_research_papers(self, paper_sources: List[str]) -> Tuple[ResearchFindings, List[str]]:
        """
        Process research papers from various sources (files or URLs).
        
        Args:
            paper_sources: List of file paths or URLs to research papers
            
        Returns:
            Tuple of (research findings, list of processed sources)
        """
        paper_texts = []
        processed_sources = []
        
        for source in paper_sources:
            try:
                # Check if source is a URL
                if source.startswith(('http://', 'https://')):
                    paper_text = self.pdf_processor.fetch_url_content(source)
                    source_name = source
                else:
                    # Assume it's a file path
                    paper_text = self.pdf_processor.process_input(source)
                    source_name = os.path.basename(source)
                
                if paper_text:
                    paper_texts.append(paper_text)
                    processed_sources.append(source_name)
            except Exception as e:
                print(f"Error processing source {source}: {e}")
        
        # Analyze all papers
        research_findings = self.nlp_analyzer.analyze_multiple_papers(paper_texts)
        
        return research_findings, processed_sources
    
    def generate_protocol_recommendation(self, patient_info: PatientInfo, research_findings: ResearchFindings) -> ProtocolRecommendation:
        """
        Generate MRI protocol recommendation based on patient info and research findings.
        
        Args:
            patient_info: Structured patient information
            research_findings: Structured research findings
            
        Returns:
            Protocol recommendation
        """
        return self.protocol_recommender.generate_recommendation(patient_info, research_findings)
    
    def process_complete_workflow(self, patient_text: str, paper_sources: List[str]) -> Dict[str, Any]:
        """
        Process the complete workflow from patient data and research papers to protocol recommendation.
        
        Args:
            patient_text: Text containing patient information
            paper_sources: List of file paths or URLs to research papers
            
        Returns:
            Dictionary with protocol recommendation and processing metadata
        """
        # Process patient data
        patient_info = self.process_patient_data(patient_text)
        
        # Process research papers
        research_findings, processed_sources = self.process_research_papers(paper_sources)
        
        # Generate recommendation
        recommendation = self.generate_protocol_recommendation(patient_info, research_findings)
        
        # Format response
        response = {
            'recommendation': {
                'sequences': recommendation.sequences,
                'field_strength': recommendation.field_strength,
                'contrast_agent': recommendation.contrast_agent or 'None (non-contrast protocol)',
                'special_considerations': recommendation.special_considerations,
                'rationale': recommendation.rationale,
                'alternative_options': recommendation.alternative_options,
                'contraindications': recommendation.contraindications
            },
            'metadata': {
                'patient_info': patient_info.model_dump(),
                'processed_sources': processed_sources,
                'research_findings_summary': {
                    'num_protocols': len(research_findings.mri_protocols),
                    'field_strengths': research_findings.field_strengths,
                    'sequences': research_findings.sequences,
                    'conditions': research_findings.conditions
                }
            }
        }
        
        return response
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            self.pdf_processor.cleanup()
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error during cleanup: {e}")


# Example usage
if __name__ == "__main__":
    # Set up a mock API key for testing
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Initialize the planner
    planner = SmartMRIPlanner()
    
    # Example patient text
    patient_text = "Patient is a 58 year old male with a history of stage 2 hypertension and eGFR of 45mL/min/1.73m2. Assess for fibrosis."
    
    # Example paper sources (in a real scenario, these would be actual files or URLs)
    paper_sources = [
        "example_paper.pdf",  # This would be a local file path
        "https://example.com/research-paper.pdf"  # This would be a URL
    ]
    
    # In a real implementation, this would process actual data
    # For demonstration, we'll just print what would be processed
    print(f"In a real implementation with a valid API key, the workflow would process:")
    print(f"Patient text: {patient_text}")
    print(f"Paper sources: {paper_sources}")
    
    # Clean up
    planner.cleanup()

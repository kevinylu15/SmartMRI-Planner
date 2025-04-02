"""
Test script for PDF Processor module

This script tests the functionality of the PDF processor module with sample files.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pdf_processor import PDFProcessor

def create_sample_pdf():
    """Create a sample PDF file for testing."""
    import fpdf
    
    # Create a temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "sample_paper.pdf")
    
    # Create PDF
    pdf = fpdf.FPDF()
    
    # Add a page
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Sample Research Paper on MRI Protocols", ln=True, align='C')
    
    # Abstract
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Abstract", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="This paper discusses advanced MRI protocols for cardiac imaging. We present a comprehensive review of current techniques and propose new methods for improved diagnosis of cardiac fibrosis.")
    
    # Introduction
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Introduction", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Magnetic Resonance Imaging (MRI) has become an essential tool in cardiac diagnostics. This paper reviews the latest developments in cardiac MRI protocols with a focus on fibrosis detection in patients with various risk factors including hypertension and reduced kidney function.")
    
    # Methods
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Methods", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="We conducted a systematic review of cardiac MRI protocols used in 15 major medical centers. The protocols were analyzed for their effectiveness in detecting cardiac fibrosis in patients with comorbidities. T1 and T2 mapping sequences at both 1.5T and 3T field strengths were evaluated.")
    
    # Results
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Results", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Our analysis found that T1 mapping at 3T provided the highest sensitivity for detecting diffuse fibrosis in patients with stage 2 hypertension. For patients with reduced kidney function (eGFR < 60mL/min/1.73m2), non-contrast protocols using native T1 and T2 mapping showed promising results without the risks associated with contrast agents.")
    
    # Conclusion
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Conclusion", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="We recommend tailored MRI protocols based on patient characteristics. For patients with hypertension and reduced kidney function, a combination of native T1 and T2 mapping at 3T with breath-held acquisitions provides optimal diagnostic value while minimizing risks.")
    
    # Output the PDF
    pdf.output(pdf_path)
    
    return pdf_path, temp_dir

def test_pdf_processor():
    """Test the PDF processor with a sample PDF file."""
    print("Testing PDF Processor module...")
    
    # Create a sample PDF
    pdf_path, temp_dir = create_sample_pdf()
    print(f"Created sample PDF at: {pdf_path}")
    
    # Initialize the PDF processor
    processor = PDFProcessor()
    
    # Test PDF extraction
    print("\nTesting PDF text extraction...")
    text = processor.process_input(pdf_path)
    print(f"Extracted {len(text)} characters of text")
    print(f"Sample of extracted text: {text[:200]}...")
    
    # Test section extraction
    print("\nTesting section extraction...")
    sections = processor.extract_sections(text)
    for section, content in sections.items():
        print(f"Section: {section}")
        print(f"Content length: {len(content)} characters")
        print(f"Sample: {content[:100]}...")
        print()
    
    # Test URL fetching with a sample URL
    print("\nTesting URL content fetching...")
    try:
        url_text = processor.fetch_url_content("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6028769/")
        print(f"Extracted {len(url_text)} characters from URL")
        print(f"Sample of URL text: {url_text[:200]}...")
    except Exception as e:
        print(f"URL fetching test failed: {e}")
    
    # Clean up
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    processor.cleanup()
    
    print("\nPDF Processor testing completed.")

if __name__ == "__main__":
    test_pdf_processor()

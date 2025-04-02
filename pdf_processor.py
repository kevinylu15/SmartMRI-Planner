"""
PDF Processing Module for SmartMRI Planner

This module handles the extraction of text from PDFs and URLs containing research articles.
It provides functionality to:
1. Extract text from local PDF files
2. Fetch and extract content from URLs (both HTML pages and PDFs)
3. Preprocess the extracted text for NLP analysis
"""

import os
import re
import tempfile
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import PyPDF2
import pdfplumber
import subprocess
from typing import Dict, List, Optional, Union


class PDFProcessor:
    """Class for processing PDFs and URLs to extract text content."""
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.temp_dir = tempfile.mkdtemp()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using multiple methods for robustness.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        text = ""
        
        # Try using PyPDF2 first
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
        
        # If PyPDF2 didn't extract much text, try pdfplumber
        if len(text.strip()) < 100:
            try:
                text = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or "" + "\n\n"
            except Exception as e:
                print(f"pdfplumber extraction failed: {e}")
        
        # If both libraries failed, try using poppler-utils (pdftotext)
        if len(text.strip()) < 100:
            try:
                output_file = os.path.join(self.temp_dir, "output.txt")
                subprocess.run(["pdftotext", pdf_path, output_file], check=True)
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
            except Exception as e:
                print(f"pdftotext extraction failed: {e}")
        
        return self._preprocess_text(text)
    
    def fetch_url_content(self, url: str) -> str:
        """
        Fetch content from a URL, handling both HTML pages and PDFs.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Extracted text from the URL
        """
        try:
            response = requests.get(url, headers={'User-Agent': 'SmartMRI-Planner/1.0'})
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Handle PDF URLs
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                temp_pdf = os.path.join(self.temp_dir, "temp.pdf")
                with open(temp_pdf, 'wb') as f:
                    f.write(response.content)
                return self.extract_text_from_pdf(temp_pdf)
            
            # Handle HTML pages
            else:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Extract text
                text = soup.get_text(separator=' ')
                
                # Clean up text
                return self._preprocess_text(text)
                
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
            return ""
    
    def process_input(self, input_source: str) -> str:
        """
        Process input source which can be either a file path or URL.
        
        Args:
            input_source: Path to PDF file or URL
            
        Returns:
            Extracted text content
        """
        # Check if input is a URL
        parsed = urlparse(input_source)
        if parsed.scheme and parsed.netloc:
            return self.fetch_url_content(input_source)
        
        # Check if input is a local file
        elif os.path.isfile(input_source):
            if input_source.lower().endswith('.pdf'):
                return self.extract_text_from_pdf(input_source)
            else:
                # Handle text files
                with open(input_source, 'r', encoding='utf-8', errors='ignore') as file:
                    return self._preprocess_text(file.read())
        else:
            raise ValueError(f"Input source not found or not supported: {input_source}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text to clean and normalize it.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
            
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        # Remove references and citations patterns like [1], [2-4], etc.
        text = re.sub(r'\[\d+(?:-\d+)?\]', '', text)
        
        return text.strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Attempt to extract common sections from academic papers.
        
        Args:
            text: Preprocessed text from a paper
            
        Returns:
            Dictionary with section names as keys and content as values
        """
        # Common section headers in academic papers
        section_patterns = [
            r'abstract', r'introduction', r'background', r'methods?', 
            r'methodology', r'results?', r'discussion', r'conclusion', 
            r'references', r'acknowledgements?'
        ]
        
        # Create a regex pattern to find these sections
        pattern = r'(?i)^(' + '|'.join(section_patterns) + r')[\s:]*$'
        
        # Find all potential section headers
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        
        sections = {}
        
        # Extract content between section headers
        for i, match in enumerate(matches):
            section_name = match.group(1).lower()
            start_pos = match.end()
            
            # If this is the last section, extract until the end of the text
            if i == len(matches) - 1:
                section_content = text[start_pos:].strip()
            else:
                end_pos = matches[i + 1].start()
                section_content = text[start_pos:end_pos].strip()
            
            sections[section_name] = section_content
        
        # If no sections were found, use the whole text as content
        if not sections:
            sections['full_text'] = text
            
        return sections
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Example with a local PDF file
    # text = processor.process_input("path/to/paper.pdf")
    
    # Example with a URL
    # text = processor.process_input("https://example.com/research-paper.pdf")
    
    # print(f"Extracted {len(text)} characters of text")
    # sections = processor.extract_sections(text)
    # for section, content in sections.items():
    #     print(f"Section: {section}")
    #     print(f"Content length: {len(content)} characters")
    
    # processor.cleanup()

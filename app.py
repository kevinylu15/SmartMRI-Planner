import os
from typing import Optional, List
#import fitz  # PyMuPDF, kept in case you want to support PDFs later
from medpalm.model import MedPalm  # Import the MedPalm model
from flask import Flask, request, render_template
from dotenv import load_dotenv
from langchain.llms import OpenAI, LLM
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader  # To load scientific papers from URL(s)
from langchain_community.document_loaders import PyMuPDFLoader  # For PDF support if needed
from langchain_community.llms import OpenAI, LLM

# Load environment variables
load_dotenv()

app = Flask(__name__)

class MedpalmLLM(LLM):
    def __init__(self, model=None):
        super().__init__()
        self.model = model or self._initialize_model()
    
    def _initialize_model(self):
        # Initialize MedPalm model here
        return MedPalm()  
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Implement actual MedPalm query logic here
            img = torch.randn(1, 3, 256, 256)  # Placeholder for actual image input
            text = torch.randint(0, 20000, (1, 4096))  # Placeholder for actual text input
            output = self.model(img, text)
            return f"MedPalm analysis: {output}"  # Placeholder for actual output processing
        except Exception as e:
            raise RuntimeError(f"MedPalm query failed: {str(e)}")

    @property
    def _identifying_params(self):
        return {"name": "MedPaLMLLM"}

def extract_text_from_webpage(url: str) -> str:
    loader = WebBaseLoader(url)
    docs = loader.load()  # Returns a list of Document objects
    # Combine text from all loaded documents (or select the most relevant one)
    combined_text = "\n".join(doc.page_content for doc in docs)
    return combined_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        clinical_query = request.form["clinical_query"]
        # Retrieve the URL input for the scientific paper(s) on MRI protocols
        paper_url = request.form.get("paper_url")
        
        try:
            document_text = ""
            
            # If a URL is provided, extract text from the scientific paper
            if paper_url:
                document_text = extract_text_from_webpage(paper_url)
            
            # Ensure that we have some document text to work with
            if not document_text.strip():
                raise ValueError("No document text could be extracted. Please provide a valid URL to a scientific paper on MRI protocols.")
            
            # Verify API key exists
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key not found in environment variables")
                
            # OpenAI LLM Chain Setup (acting as an MRI protocol expert)
            openai_llm = OpenAI(temperature=0.7)
            openai_prompt_template = PromptTemplate(
                input_variables=["clinical_query", "document_text"],
                template=(
                    "You are an expert in MRI protocols. Given the clinical prompt below and the scientific paper text on MRI protocols, "
                    "analyze the information and provide a concise recommendation with details such as sequences, field strength, and special considerations.\n\n"
                    "Clinical Prompt: {clinical_query}\n\n"
                    "Scientific Paper Text: {document_text}\n\n"
                    "Recommendation (e.g., Sequences, Field Strength, Special Considerations):"
                )
            )
            openai_chain = LLMChain(llm=openai_llm, prompt=openai_prompt_template)

            # MedPaLM LLM Chain Setup to add additional insights or edge cases
            medpalm_llm = MedpalmLLM()
            medpalm_prompt_template = PromptTemplate(
                input_variables=["openai_response"],
                template=(
                    "Review the following recommendation from OpenAI and enhance it by adding any additional insights, particularly "
                    "addressing rare conditions or edge cases:\n\n"
                    "{openai_response}\n\n"
                    "Enhanced Recommendation:"
                )
            )
            medpalm_chain = LLMChain(llm=medpalm_llm, prompt=medpalm_prompt_template)

            # Combine the Chains Sequentially
            overall_chain = SimpleSequentialChain(chains=[openai_chain, medpalm_chain], verbose=True)

            # Run the Combined Chain with the Clinical Prompt and Scientific Paper Text
            final_response = overall_chain.run(clinical_query=clinical_query, document_text=document_text)
            return render_template("index.html", response=final_response)
        except Exception as e:
            return render_template("index.html", error=str(e))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
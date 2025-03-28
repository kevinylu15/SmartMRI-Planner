## **SmartMRI Planner**

SmartMRI Planner is a web-based application built in Python that leverages advanced language models to optimize MRI protocol recommendations. Unlike traditional statistical approaches, this tool uses a sequential LLM chain—first consulting an OpenAI‑powered model to interpret clinical queries alongside scientific literature, and then enhancing the output with specialized insights via a MedPalm model.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Explanation](#algorithm-explanation)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)


## **Introduction**

SmartMRI Planner is designed to assist clinicians and researchers in planning MRI protocols by combining expert-level language processing with cutting-edge machine learning. The application uses two LLM chains:

OpenAI LLM Chain: Acts as an MRI protocol expert by analyzing clinical prompts and text extracted from scientific papers.

MedPalm LLM Chain: Further refines the initial recommendation by addressing rare conditions and edge cases.

This dual‑chain setup is implemented using the LangChain library and Flask for a responsive web interface.


## **Features**

Web Interface: Built using Flask to allow users to input clinical queries and scientific paper URLs.

Dynamic Document Loading: Utilizes WebBaseLoader to extract text from online scientific articles.

Sequential LLM Processing:

The first chain uses an OpenAI‑based model to generate an initial recommendation based on clinical context and scientific literature.

The second chain employs a MedPalm model to enhance the recommendation with additional insights.

Flexible Configuration: Environment variables (e.g., OPENAI_API_KEY) manage API keys and settings.

Extensibility: The code is structured to allow future enhancements, such as PDF support via PyMuPDF or more robust MedPalm integration.

## **Installation**

To install, clone the repository, create a virtual environment, install dependencies, and setup environment variables to run the application.

## **Usage**

Access the Web Interface: Open your browser and navigate to the provided URL.

Input Your Data:

Clinical Query: Enter your clinical question or scenario regarding MRI protocols.

Scientific Paper URL: Provide a URL linking to a relevant scientific paper (optional, but recommended for more detailed insights).

Submit: The system will extract the paper’s content, run the dual LLM chain to analyze the input, and then display a detailed recommendation.

Receive Recommendations: The final output includes proposed MRI sequences, suggested field strengths, and special considerations—including insights on rare conditions.

## **Algorithm Explanation**

SmartMRI Planner processes requests in two main stages:

OpenAI LLM Chain:

Input: Clinical prompt and text extracted from a scientific paper.

Task: Analyze and provide a concise recommendation covering sequences, field strength, and other considerations.

Prompt Example:

pgsql
Copy
You are an expert in MRI protocols. Given the clinical prompt below and the scientific paper text on MRI protocols, analyze the information and provide a concise recommendation with details such as sequences, field strength, and special considerations.
MedPalm LLM Chain:

Input: The recommendation generated by the OpenAI chain.

Task: Enhance the recommendation by adding further insights and addressing potential edge cases or rare conditions.

Prompt Example:

csharp
Copy
Review the following recommendation from OpenAI and enhance it by adding any additional insights, particularly addressing rare conditions or edge cases:
Sequential Execution:

The output of the first chain becomes the input for the second, ensuring that both general expertise and specialized considerations are addressed.

The final recommendation is then rendered on the web interface.

## **Limitations**

Error Handling: The function has basic error checking for formulas and mathematical operations but may not handle all edge cases.
Extensions: Only supports binary response variables.
Performance: Potentially slow for large, complex datasets.
Testing: Does not display significance test results or standard errors.

## **Contributing**

If you'd like to improve the function or add features, please submit a pull request or send me an email.

## **License**

The Logistic Regression Model Funciton is released under the MIT License. See the **[LICENSE](https://www.blackbox.ai/share/LICENSE)** file for details.
# AI-Powered-PDF-to-PowerPoint-Convertor
program that utilises AI and existing software. The program will allow a user to upload a PDF which depicts, text, graphs and tables regarding a company (company A), from which the program outputs a complete copy, however for (specified) company B.

I want the program to utilise Marker, Surya, Index Lllama and other pdf converters, use AI in to decipher the differing responses and select the most accurate output.

I want the program to be able to utilise Capital IQ to output financial information for company B (the specific information will be specified by the PDF slides of Company A).

So far i have thought that the the program should:

The primary objective of this program is to create a comprehensive document processing system that can:
1. Extract and structure content from PDF documents
2. Provide an intelligent query interface for the processed documents
3. Recreate the PDF content in PowerPoint presentations
4. Update financial data in the presentation using Capital IQ data
Key Processes:
1. PDF Input and Analysis:
   - Accept a PDF file input
   - Determine if OCR is needed
2. Content Extraction:
   - For OCR-needed documents: Convert to images and use multiple OCR methods (Tesseract with different configurations)
   - For non-OCR documents: Extract text directly
   - Extract tables and images from all documents
3. Content Processing:
   - Resolve discrepancies in OCR results
   - Implement human verification for low-confidence results
4. Structured Output Creation:
   - Organize content into a JSON format
   - Convert tables to CSV
   - Convert images to base64
   - Store metadata (including image positions)
5. Indexing and Querying:
   - Use a custom embedding model (based on DistilBERT)
   - Create a VectorStoreIndex using llama_index
   - Implement a custom postprocessor for metadata inclusion in query responses
6. PowerPoint Recreation:
   - Create slides matching the original PDF layout
   - Place text, tables, and images in appropriate positions
   - Maintain original formatting as closely as possible
7. Capital IQ Integration:
   - Identify financial data points in the content
   - Map data points to Capital IQ API endpoints
   - Fetch updated financial data
   - Replace original data with updated information
8. Dynamic Content Update:
   - Update tables with new financial data
   - Regenerate graphs and charts
   - Adjust text content to reflect new information
9. User Interaction:
   - Provide an interactive query interface
   - Allow selection of company for data updates
   - Offer options for data verification
Key Priorities:
1. Accuracy and Comprehensiveness:
   - Ensure accurate extraction of text, tables, and images
   - Preserve document structure and layout information
2. Efficient Data Handling:
   - Implement metadata storage to manage large documents
   - Use efficient indexing for quick query responses
3. Flexibility and Robustness:
   - Handle various PDF types (text-based, image-based, mixed)
   - Adapt to different document structures and layouts
4. PowerPoint Fidelity:
   - Accurately recreate PDF content in PowerPoint
   - Maintain original design and layout as closely as possible
5. Dynamic Financial Data Integration:
   - Seamlessly integrate Capital IQ data
   - Accurately update financial information across the presentation
6. User-Friendly Experience:
   - Provide clear prompts and information during processing
   - Offer an intuitive query interface
   - Allow easy selection of companies for data updates
7. Data Integrity and Verification:
   - Implement checks for data consistency
   - Provide user verification options for significant changes
8. Performance and Scalability:
   - Optimize for handling large, complex documents
   - Balance processing thoroughness with speed
9. Extensibility:
   - Structure code for easy addition of new features
   - Allow for future integration of additional data sources or presentation formats
This program aims to transform static PDF financial documents into dynamic, queryable, and updatable PowerPoint presentations. It addresses the challenges of complex document processing, intelligent information retrieval, and the need for up-to-date financial data in presentations. The system prioritizes accuracy, user experience, and the flexibility to handle diverse document types and financial data points.
========================
Creating an AI-powered PDF document converter that extracts and processes financial data from company-specific documents, then updates it with information from external sources like Capital IQ, and finally converts this into a PowerPoint presentation, is an ambitious but feasible project. Below is a breakdown of how to approach building this system, along with some code snippets and recommendations for each step.
High-Level Architecture

The system can be broken down into the following modules:

    PDF Input and Analysis
    Content Extraction
    Content Processing and Transformation
    Indexing and Querying
    PowerPoint Recreation
    Capital IQ Integration
    User Interaction

Key Steps and Code Snippets
1. PDF Input and Analysis

The first step is to process the uploaded PDF. This involves:

    Determining whether OCR is required (i.e., whether the document is image-based or text-based).
    Extracting text, tables, and images from PDFs using libraries like PyPDF2, pdfminer, or pdfplumber for text extraction, and Tesseract for OCR if needed.

import pytesseract
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pdfplumber

def extract_text_from_pdf(pdf_path):
    # Attempt to extract text using pdfplumber (for text-based PDFs)
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def ocr_pdf(pdf_path):
    # If OCR is required, convert PDF to images and apply Tesseract OCR
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def process_pdf(pdf_path):
    # Check if OCR is needed (You could add more logic for this step)
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():  # If no text is found, use OCR
        text = ocr_pdf(pdf_path)
    return text

2. Content Extraction and Structuring

Once the text is extracted, the next step is to identify and structure it, especially when dealing with tables and images. You will also need to handle discrepancies in OCR results and ensure that extracted tables are in CSV format.

import pandas as pd

def extract_tables_from_pdf(pdf_path):
    # Using pdfplumber to extract tables
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            for table in page.extract_tables():
                tables.append(pd.DataFrame(table[1:], columns=table[0]))
    return tables

def extract_images_from_pdf(pdf_path):
    # Extract images using pdfminer or similar libraries
    # For now, we just return base64 representation as a placeholder
    return ["image_base64_data"]  # Replace with actual image extraction code

3. Content Processing and Transformation

You’ll need to process the extracted content. This includes using AI models for mapping content between Company A and Company B, ensuring text and tables are converted correctly, and any necessary adjustments to tables, graphs, or charts.

For this, you can use LlamaIndex (previously called GPT Index) for indexing and querying extracted information and Transformer models like DistilBERT to understand the content and structure for comparison.

from llama_index import VectorStoreIndex, SimpleKeywordTableIndex
from transformers import DistilBertTokenizer, DistilBertModel

# Use a transformer model for mapping content (company A to company B)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def query_content_for_mapping(content, company_a_data, company_b_data):
    # This function would utilize the AI model to find content related to company A
    # and replace it with company B's information
    inputs = tokenizer(content, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    # Further logic here to map content from company A to company B
    return content.replace(company_a_data, company_b_data)  # Basic replacement for now

4. PowerPoint Recreation

To create PowerPoint slides, you can use the python-pptx library, which allows you to add text, tables, and images to slides. You will need to replicate the layout and style of the original PDF content.

from pptx import Presentation
from pptx.util import Inches

def create_presentation(extracted_data, tables, images):
    prs = Presentation()

    # Add a slide for text content
    slide = prs.slides.add_slide(prs.slide_layouts[1])  # Title and Content layout
    title = slide.shapes.title
    title.text = "Company Presentation"

    content_box = slide.shapes.placeholders[1]
    content_box.text = extracted_data  # Add the extracted text to the slide

    # Add tables to slides
    for table in tables:
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Blank layout
        rows, cols = len(table), len(table.columns)
        table_shape = slide.shapes.add_table(rows + 1, cols, Inches(1), Inches(1), Inches(7), Inches(4))
        table = table_shape.table

        # Set column names
        for col_num, column_name in enumerate(table.columns):
            table.cell(0, col_num).text = column_name

        # Add table data
        for row_num, row_data in enumerate(table.values):
            for col_num, cell_data in enumerate(row_data):
                table.cell(row_num + 1, col_num).text = str(cell_data)

    # Add images
    for img_base64 in images:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.add_picture(img_base64, Inches(1), Inches(1), Inches(4), Inches(3))  # Adjust dimensions accordingly

    prs.save("Company_B_Presentation.pptx")

5. Capital IQ Integration

To fetch financial data from Capital IQ, you’ll need to access their API (this requires an API key). Here’s an example of how to fetch data using the requests library. Ensure you are adhering to the Capital IQ API's documentation.

import requests

def fetch_capital_iq_data(company_symbol):
    url = f"https://api.capitaliq.com/v1/companies/{company_symbol}/financials"
    headers = {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json',
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    return data  # Process the returned data as needed

6. Dynamic Content Update

Once you’ve fetched financial data from Capital IQ, you need to dynamically update the tables, charts, and other financial details in the PowerPoint presentation.

def update_financial_data_in_presentation(presentation_path, company_data):
    prs = Presentation(presentation_path)

    # Search through the slides and update financial information
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, 'text') and "financial data placeholder" in shape.text:
                shape.text = shape.text.replace("old_value", company_data["new_value"])

    prs.save("Updated_Company_B_Presentation.pptx")

Key Steps for Building the Full System

    Document Preprocessing:
        Handle OCR for scanned documents.
        Use PDF extraction tools to gather content (text, tables, images).

    AI and Model Integration:
        Use AI models like DistilBERT to map company-specific data between Company A and Company B.
        Use LlamaIndex to index content and make it queryable.

    PowerPoint Creation:
        Use python-pptx to generate slides from structured content.
        Ensure that formatting is preserved as much as possible when transferring content from PDF to PowerPoint.

    Data Retrieval:
        Integrate with Capital IQ API to fetch live financial data and replace placeholders in the PowerPoint.

    User Interface:
        Provide a web-based or command-line interface for users to upload PDFs and specify Company B.

Conclusion

This solution provides a comprehensive, AI-powered PDF-to-PowerPoint converter that handles content extraction, data transformation, and updates financial information using external APIs like Capital IQ. The main challenge is to ensure the accuracy of the data transformation and presentation recreation, as well as the seamless integration of external data sources for financial updates.



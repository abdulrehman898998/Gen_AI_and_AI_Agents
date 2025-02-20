# AI-Powered Presentation Generator

## Overview
This is a Streamlit-based web application that generates AI-powered presentations. Users can enter a topic, specify the number of slides, and optionally upload a document for content extraction. The app integrates with local (Ollama) and cloud-based (OpenRouter) LLMs to generate structured presentation slides. Additionally, it fetches relevant images from the Pexels API and allows users to download the presentation as a PDF.

## Features
- Generate AI-powered presentations with customizable topics and slide counts.
- Option to upload a document for content-driven slide generation.
- Supports both local (Ollama) and cloud-based (OpenRouter) LLMs.
- Uses Pexels API to fetch high-quality, royalty-free images for slides.
- Provides a PDF export feature for generated slides.

## Installation
### Prerequisites
- Python 3.8+
- Pip

### Install Dependencies
Run the following command to install required packages:
```sh
pip install streamlit requests jsonschema pydantic reportlab Pillow openai langchain_community
```

## API Keys
This application requires an API key for fetching images from Pexels.
- Set your Pexels API key in the `PEXELS_API_KEY` variable inside the script.
- If using OpenRouter, provide the API key in the Streamlit UI when selecting the cloud model.

## Running the Application
Run the following command in your terminal:
```sh
streamlit run your_script.py
```
Replace `your_script.py` with the actual filename.

## Usage
1. Open the Streamlit web app.
2. Configure the model settings in the sidebar (choose local or cloud LLM and enter the API key if needed).
3. Enter a topic for the presentation.
4. Specify the number of slides.
5. (Optional) Upload a document to provide additional content.
6. Click "Generate Presentation" and wait for the AI-generated slides.
7. Download the presentation as a PDF.

## Core Functions
- **handle_llm_response(response):** Processes responses from LLMs.
- **robust_json_parser(raw):** Extracts valid JSON from raw AI-generated text.
- **fetch_pexels_image(query):** Fetches images from Pexels API based on search query.
- **generate_pdf_presentation(slides):** Generates a downloadable PDF presentation.

## Technologies Used
- **Streamlit** - Web application framework for Python.
- **Pydantic** - Data validation for structured slide generation.
- **ReportLab** - PDF generation.
- **OpenAI API** - Cloud-based AI model support (optional).
- **Ollama** - Local LLM execution (optional).
- **Pexels API** - Fetching relevant images.

## License
This project is open-source. Feel free to modify and use it as needed.

## Future Enhancements
- Support for additional LLM providers.
- Improved slide layout customization.
- Enhanced UI with drag-and-drop content editing.

## Contributions
Pull requests and feature suggestions are welcome!


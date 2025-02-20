import streamlit as st
import requests
import json
import re
import time
from typing import List, Optional
from io import BytesIO
from openai import OpenAI
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field, ValidationError

# Additional libraries for PDF generation.
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image

# ========== CONSTANTS ==========
PEXELS_API_KEY = 'r7OR9sOr3eZsDinM1UARYUsTyYUDk8A3rSoHl1WQrQ0SfWE8d5VLVxCR'
DEFAULT_TOPIC = "AI in Modern Healthcare"

# ========== DATA MODELS ==========
class Slide(BaseModel):
    title: str = Field(..., max_length=80)
    content: List[str] = Field(..., min_items=4, max_items=6)
    image_query: str = Field(..., min_length=2)
    image_url: Optional[str] = None

# ========== CORE FUNCTIONS ==========
def handle_llm_response(response) -> str:
    """Universal LLM response handler"""
    if response is None:
        return ""
    if hasattr(response, 'content'):
        return response.content
    return str(response)

def robust_json_parser(raw: str) -> dict:
    """
    Enhanced JSON parser with multiple extraction and cleaning strategies.
    This version avoids recursive regex constructs.
    """
    def attempt_parse(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # Strategy 1: Try parsing the full raw string.
    parsed = attempt_parse(raw)
    if parsed is not None:
        return parsed

    # Strategy 2: Trim to the first '{' and the last '}'.
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        parsed = attempt_parse(candidate)
        if parsed is not None:
            return parsed

    # Strategy 3: Use a simple regex to capture a JSON-like block.
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        candidate = match.group(0)
        # Clean candidate: remove characters that are unlikely in JSON.
        cleaned = re.sub(r'[^{}\[\]:,"0-9a-zA-Z_ \-\.]', '', candidate)
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        parsed = attempt_parse(cleaned)
        if parsed is not None:
            return parsed

    snippet = raw[:500]  # Provide a snippet for debugging.
    raise ValueError(f"No valid JSON structure found.\nRaw response snippet:\n{snippet}")

def generate_pdf_presentation(slides: List[Slide]) -> BytesIO:
    """
    Generate a PDF presentation where each slide is a separate page.
    Uses ReportLab to draw text and images.
    """
    buffer = BytesIO()
    # Use a landscape letter page size.
    page_width, page_height = landscape(letter)
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    
    for slide in slides:
        # 1. Draw the Title
        pdf.setFont("Helvetica-Bold", 32)
        pdf.drawString(50, page_height - 70, slide.title)
        
        # 2. Draw bullet point content
        pdf.setFont("Helvetica", 18)
        bullet_y = page_height - 120
        line_spacing = 30
        
        for point in slide.content:
            pdf.drawString(70, bullet_y, u'\u2022 ' + point)
            bullet_y -= line_spacing

        # Add some extra space between text and image
        bullet_y -= 30  # Adjust this for more/less space
        
        # 3. If an image URL exists, try to download and embed the image in the center
        if slide.image_url:
            try:
                response = requests.get(slide.image_url, timeout=10)
                response.raise_for_status()
                img_data = BytesIO(response.content)
                image = Image.open(img_data).convert("RGB")
                
                # Make the image slightly larger, up to 400×300
                max_width, max_height = 400, 300
                image.thumbnail((max_width, max_height))  # Maintains aspect ratio
                final_width, final_height = image.size
                
                # Calculate centered X position
                x_pos = (page_width - final_width) / 2
                # Place the image below the bullet points
                # Add an extra 20 px spacing below bullet points if desired
                y_pos = bullet_y - final_height - 20

                # Draw the image
                pdf.drawImage(ImageReader(image), x_pos, y_pos,
                              width=final_width, height=final_height)
            except Exception as e:
                st.warning(f"Image load error for slide '{slide.title}': {str(e)}")
        
        # 4. Move to the next page for the next slide
        pdf.showPage()
    
    pdf.save()
    buffer.seek(0)
    return buffer



def fetch_pexels_image(query: str) -> Optional[str]:
    """Fetch CC-licensed images from Pexels."""
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 1, "orientation": "landscape"},
            timeout=15
        )
        # Check if the response is OK.
        if response.status_code != 200:
            st.warning(f"Pexels API returned status code {response.status_code}: {response.text}")
            return None
        data = response.json()
        photos = data.get('photos')
        if photos and len(photos) > 0:
            return photos[0]['src']['large']
        else:
            st.warning("No photos found for query: " + query)
            return None
    except Exception as e:
        st.warning(f"Pexels API Error: {str(e)}")
        return None

# ========== STREAMLIT UI ==========
def main():
    st.set_page_config(page_title="AI Presentation Builder", page_icon="📊", layout="wide")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        llm_type = st.radio("Model Type", ["Local (Ollama)", "Cloud (OpenRouter)"], index=1)
        model_name = st.text_input("Model Name", value="deepseek-r1" if llm_type == "Local (Ollama)" else "deepseek/deepseek-r1:free")
        api_key = st.text_input("API Key", type="password", help="Required for OpenRouter models", disabled=(llm_type != "Cloud (OpenRouter)"))
    
    st.title("🚀 AI-Powered Presentation Generator")
    
    # Option for the user to upload a document.
    st.subheader("Upload a Document (Optional)")
    uploaded_file = st.file_uploader("Upload a text file (.txt) with your document content", type=["txt"])
    document_text = ""
    if uploaded_file is not None:
        try:
            document_text = uploaded_file.read().decode("utf-8")
            st.info("Document uploaded successfully. The LLM will use this content to generate the presentation.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    with st.form("presentation_form"):
        topic = st.text_input("Presentation Topic", DEFAULT_TOPIC)
        num_slides = st.slider("Number of Slides", 3, 20, 6)
        submitted = st.form_submit_button("Generate Presentation")
    
    if submitted:
        try:
            if llm_type == "Cloud (OpenRouter)" and not api_key:
                raise ValueError("API key required for cloud models")
            
            # Initialize the appropriate LLM.
            if llm_type == "Local (Ollama)":
                llm = Ollama(model=model_name, temperature=0.6)
            else:
                llm = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            
            with st.status("Generating presentation...", expanded=True):
                # Modify prompt based on whether a document was uploaded.
                if document_text:
                    prompt = f"""
Based on the following document:
-----------------------
{document_text}
-----------------------
Generate exactly {num_slides} slides about "{topic}" in JSON format.

Ensure:
1. Output starts with '{{' and ends with '}}'
2. Use double quotes (") for all strings
3. No markdown, special characters, or explanations
4. Each slide's "title" must be short (5 words or fewer).
5. Each slide's "image_query" must be unique, descriptive, and different from the other slides to avoid fetching similar images.

Example:
{{
  "slides": [
    {{
      "title": "Short Title",
      "content": ["Point 1", "Point 2", "Point 3", "Point 4"],
      "image_query": "AI revolution with robots"
    }}
  ]
}}
    """
                else:
                    prompt = f"""
                Generate exactly {num_slides} slides about "{topic}" in JSON format.

                Ensure:
                1. Output starts with '{{' and ends with '}}'
                2. Use double quotes (") for all strings
                3. No markdown, special characters, or explanations
                4. Each slide's "title" must be short (5 words or fewer).
                5. Each slide's "image_query" must be unique, descriptive, and different from the other slides to avoid fetching similar images.

                Example:
                {{
                "slides": [
                    {{
                    "title": "Short Title",
                    "content": ["Point 1", "Point 2", "Point 3", "Point 4"],
                    "image_query": "AI revolution with robots"
                    }}
                ]
                }}
                    """
                # Get raw response from the LLM.
                if llm_type == "Local (Ollama)":
                    raw_response = llm.invoke(prompt)
                else:
                    completion = llm.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "system", "content": prompt}],
                        temperature=0.6
                    )
                    if not completion or not hasattr(completion, "choices") or len(completion.choices) == 0:
                        raise ValueError("LLM returned an invalid response.")
                    raw_response = completion.choices[0].message.content

                processed_response = handle_llm_response(raw_response)
                if not processed_response:
                    raise ValueError("LLM returned an empty response.")

                st.write("### Raw LLM Response:")
                st.code(processed_response, language="json")

                # Parse the JSON using robust methods.
                parsed_data = robust_json_parser(processed_response)

                # Safely extract slides data.
                slides_data = parsed_data.get("slides")
                if not isinstance(slides_data, list):
                    raise ValueError("Parsed JSON does not contain a valid 'slides' list.")
                slides_data = slides_data[:num_slides]

                # Ensure each slide's content has at least 4 items.
                for slide in slides_data:
                    content_list = slide.get("content", [])
                    if not isinstance(content_list, list):
                        content_list = []
                    if len(content_list) < 4:
                        missing = 4 - len(content_list)
                        for i in range(missing):
                            content_list.append(f"Additional point {i+1}")
                    slide["content"] = content_list[:6]

                slides = [Slide(**s) for s in slides_data]
                st.write("### Fetched Slides Data:")
                st.json([slide.dict() for slide in slides])

                # Fetch images for each slide.
                st.write("🔍 Fetching relevant images...")
                progress_bar = st.progress(0)
                for idx, slide in enumerate(slides):
                    slide.image_url = fetch_pexels_image(slide.image_query)
                    # Use max(1, len(slides)) to avoid division by zero.
                    progress_bar.progress((idx + 1) / max(1, len(slides)))
                    time.sleep(0.2)

                # Generate PDF presentation.
                st.write("📊 Building PDF presentation...")
                pdf_file = generate_pdf_presentation(slides)
                
                st.download_button(
                    label="💾 Download Presentation (PDF)",
                    data=pdf_file,
                    file_name="ai_presentation.pdf",
                    mime="application/pdf"
                )

        except (ValidationError, ValueError) as e:
            st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Generation Failed: {str(e)}")

if __name__ == "__main__":
    main()

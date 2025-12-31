import base64
import io
import os
import json
import requests
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import TableItem, TextItem, PictureItem, SectionHeaderItem, ListItem

class PDFParser:
    """
    A comprehensive PDF parser that extracts structured content including text, tables, and images,
    and merges consecutive blocks across pages.
    """
    def __init__(self, min_width=200, min_height=200, min_area=1000, 
                 ollama_model="gemma3:latest", ollama_img_summarizer_model="moondream:v2",
                 ollama_url="http://localhost:11434/api/generate"):

        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.ollama_img_summarizer_model = ollama_img_summarizer_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url  # ✅ NOW INITIALIZED
        self.image_prompt = (
            "Analyze this image. "
            "1. Identify the type of image (e.g., photograph, chart, diagram, screenshot)."
            "2. Describe the visual content and list key entities or objects."
            "3. Give a detailed description and transcribe any visible text, titles, or data points."
        )
    
    @staticmethod
    def _image_to_base64(pil_image):
        """Helper to convert PIL Image to Base64 string for JSON output"""
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    @staticmethod
    def _to_b64_payload(data_uri_or_b64: str) -> str:
        """Remove 'data:image/...;base64,' prefix and clean whitespace/newlines."""
        s = (data_uri_or_b64 or "").strip()
        if s.startswith("data:image"):
            s = s.split(",", 1)[1]
        return "".join(s.split())
    
    def _describe_base64_image(self, b64_img: str) -> str:
        """Send base64 image to Ollama and return the generated description."""
        payload = {
            "model": self.ollama_img_summarizer_model,
            "prompt": self.image_prompt,
            "images": [self._to_b64_payload(b64_img)],
            "stream": False
        }
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except Exception as e:
            print(f"Error generating image description: {e}")
            return f"[Error generating description: {e}]"
        
    def _add_image_descriptions(self, data):
        """Attach image descriptions to single or multiple image entries in the data."""
        for item in data:
            if item.get("content_type") == "image":
                pc = item.get("page_content")
                
                if isinstance(pc, list):
                    for idx, obj in enumerate(pc, start=1):
                        if isinstance(obj, dict) and "base64" in obj:
                            print(f"Generating description for image {idx} on page {item.get('page_no')}...")
                            b64 = obj.get("base64")
                            if b64:
                                obj["image_description"] = self._describe_base64_image(b64)                          
        return data
    
    def extract_structured_json(self, file_path: str):
        """ Extract structured JSON content from a PDF file. """
        # 1. Configuration
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.generate_picture_images = True

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

            print(f"Processing {file_path}...")
            doc = converter.convert(file_path).document
            
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            raise

        output_data = []
        
        # State variables for grouping
        current_page = -1
        current_type = None
        current_content_buffer = []

        # 2. Helper to flush the buffer
        def flush_buffer():
            if not current_content_buffer:
                return
            
            # Join content based on type
            if current_type == "text":
                final_content = "\n\n".join(current_content_buffer)
            elif current_type == "table":
                final_content = "\n".join(current_content_buffer)
            elif current_type == "image":
                # Always keep images as a list for consistency
                final_content = current_content_buffer
            else:
                final_content = ""

            # Only add non-empty blocks
            if current_type == "text" and not final_content.strip():
                return
            if current_type == "image" and not current_content_buffer:
                return

            output_data.append({
                "page_no": current_page,
                "content_type": current_type,
                "page_content": final_content
            })

        # 3. Iterate through all items
        try:
            for item, level in doc.iterate_items():
                # Determine Item Type
                item_type = "unknown"
                item_page = -1
                
                # Get Page Number safely
                if hasattr(item, "prov") and item.prov:
                    item_page = item.prov[0].page_no
                
                if isinstance(item, TableItem):
                    item_type = "table"
                elif isinstance(item, PictureItem):
                    item_type = "image"
                elif isinstance(item, (TextItem, SectionHeaderItem, ListItem)):
                    # Ignore empty whitespace items
                    if not item.text.strip():
                        continue
                    item_type = "text"
                else:
                    continue  # Skip unknown types

                # Trigger new block if page or type changes
                if item_page != current_page or item_type != current_type:
                    flush_buffer()
                    
                    # Reset state
                    current_page = item_page
                    current_type = item_type
                    current_content_buffer = []

                # Process Content by Type
                if item_type == "table":
                    try:
                        df = item.export_to_dataframe()
                        if df is not None and not df.empty:
                            # html_table = df.to_html(index=False, border=1)
                            # markdown_table = df.to_markdown(index=False)
                            # csv_table = df.to_csv(index=False)
                            markdown_table = df.to_markdown(index=False)
                            current_content_buffer.append(markdown_table)
                        else:
                            print(f"Empty table on page {item_page}, skipping")
                    except Exception as e:
                        print(f"Table processing error on page {item_page}: {str(e)}")
                        
                elif item_type == "image":
                    try:
                        img_obj = item.get_image(doc)
                        if img_obj:
                            width, height = img_obj.size
                            
                            # Filter by size
                            if width >= self.min_width and height >= self.min_height and (width * height) >= self.min_area:
                                
                                # Create base64 string
                                b64_str = self._image_to_base64(img_obj)
                                
                                img_data = {
                                    "base64": f"data:image/png;base64,{b64_str}",
                                    "width": width,
                                    "height": height,
                                    "page": item_page
                                }
                                current_content_buffer.append(img_data)
                                
                                print(f"Extracted image: {width}x{height}px on page {item_page}")
                            else:
                                print(f"Skipping small image: {width}x{height}px on page {item_page}")
                    except Exception as e:
                        print(f"Image processing error on page {item_page}: {str(e)}")
                        
                elif item_type == "text":
                    # CHANGED: Detailed Markdown formatting logic
                    clean_text = item.text.strip()
                    
                    if isinstance(item, SectionHeaderItem):
                        # Use the 'level' attribute if available, default to 2
                        header_level = getattr(item, 'level', 2) 
                        prefix = "#" * header_level
                        current_content_buffer.append(f"{prefix} {clean_text}")
                        
                    elif isinstance(item, ListItem):
                        # Add bullet point for list items
                        current_content_buffer.append(f"* {clean_text}")
                        
                    elif isinstance(item, TextItem):
                        # Standard text
                        current_content_buffer.append(clean_text)

            # Final flush
            flush_buffer()

            # Sort by page number for consistent output
            output_data.sort(key=lambda x: x['page_no'])

        except Exception as e:
            print(f"Error during document iteration: {str(e)}")
            raise
            
        return output_data
    
    def merge_cross_page_blocks(self, data):
        """ Merge consecutive content blocks of the same type across pages."""
        if not data:
            print("Empty data")
            return []
        
        merged_data = []
        i = 0
        
        while i < len(data):
            current_block = data[i].copy()
            start_page = current_block['page_no']
            last_page = start_page
            
            # Look ahead to see if next block has same content_type
            while i + 1 < len(data):
                next_block = data[i + 1]
                
                # Check if content types match
                if current_block['content_type'] == next_block['content_type']:
                    # Track the last page number
                    last_page = next_block['page_no']
                    
                    # Merge content based on type
                    if current_block['content_type'] == 'text':
                        current_block['page_content'] += "\n\n" + next_block['page_content']
                    elif current_block['content_type'] == 'table':
                        current_block['page_content'] += "\n" + next_block['page_content']
                    elif current_block['content_type'] == 'image':
                        current_block['page_content'].extend(next_block['page_content'])
                    
                    i += 1  # Skip the merged block
                else:
                    break  # Content types don't match, stop merging
            
            # Set the final page_no range
            if start_page == last_page:
                current_block['page_no'] = start_page
            else:
                current_block['page_no'] = f"{start_page}-{last_page}"
            
            merged_data.append(current_block)
            i += 1
        
        print(f"Merged {len(data)} blocks into {len(merged_data)} blocks")
        return merged_data
    
    def parse_pdf(self, file_path: str, output_file: str = None, merge_blocks: bool = True, skip_image_descriptions: bool = False):
        """
        Main method to parse PDF and extract structured content with optional merging and image descriptions.
        
        Args:
            file_path: Path to the PDF file
            output_file: Optional path to save intermediate JSON
            merge_blocks: Whether to merge consecutive blocks of same type
            skip_image_descriptions: If True, skip image description generation (for faster processing)
        
        Returns:
            Processed data as list of dictionaries
        """

        # Get the filename from the path (e.g., "document.pdf")
        file_name = os.path.basename(file_path)

        # Extract content
        print("Step 1: Extracting content from PDF...")
        extracted_data = self.extract_structured_json(file_path)
        print(f"✓ Extracted {len(extracted_data)} content blocks")
        
        # Optionally merge blocks
        if merge_blocks:
            print("Step 2: Merging cross-page blocks...")
            final_data = self.merge_cross_page_blocks(extracted_data)
            print(f"✓ Merged into {len(final_data)} blocks")
        else:
            final_data = extracted_data
        
        # Add image descriptions (optional)
        if not skip_image_descriptions:
            print("Step 3: Generating image descriptions...")
            final_data = self._add_image_descriptions(final_data)
            print("✓ Image descriptions complete")
        else:
            print("Step 3: Skipping image descriptions (as requested)")

        # Add metadata
        print("Step 4: Adding metadata...")
        for item in final_data:
            item["title"] = file_name
             
            if '.' in file_name:
                item["category"] = file_name.split('.')[-1].lower()
            else:
                item["category"] = "unknown"
            
            page_num = str(item.get("page_no", ""))
            item["sourcepage"] = f"{file_name}::{page_num}"
        print("✓ Metadata added")
        
        # Save to file if specified
        if output_file:
            print(f"Step 5: Saving intermediate JSON to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)
            print(f"✓ Output saved to: {output_file}")
        
        print(f"✓ Extraction Complete. Processed {len(final_data)} content blocks.")
        return final_data

if __name__ == "__main__":
    parser = PDFParser(ollama_url="http://localhost:11434/api/generate")
    pdf_path = r"/Users/santusahoo/Documents/DAGENT/1177810199598.pdf"
    parser.parse_pdf(pdf_path, output_file="merged_output.json")
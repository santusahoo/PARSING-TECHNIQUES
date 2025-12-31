import json
import os
import uuid
from langchain_text_splitters import MarkdownTextSplitter


class DocumentChunker:
    def __init__(self, chunk_size=300, chunk_overlap=80, max_table_rows=1):
        """
        Initialize the processor with configuration for text splitting and table chunking.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_table_rows = max_table_rows
        
        # Initialize the LangChain splitter once
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def _chunk_markdown_table(self, markdown_text):
        """
        Internal method: Splits markdown tables into smaller chunks, preserving headers.
        """
        if not markdown_text:
            return []

        # Extract only table rows (lines starting with |)
        rows = [line.strip() for line in markdown_text.strip().split("\n") if line.strip().startswith("|")]
        
        # Validation: If not enough rows for header + separator, return as is
        if len(rows) < 2:
            return [markdown_text]
        
        # Keep the header and alignment line
        header = rows[:2]
        data_rows = rows[2:]
        
        # If table has headers but no data rows, return just the header
        if not data_rows:
            return ["\n".join(header)]

        chunks = []
        for i in range(0, len(data_rows), self.max_table_rows):
            chunk_rows = header + data_rows[i:i + self.max_table_rows]
            chunk_text = "\n".join(chunk_rows)
            chunks.append(chunk_text)
        
        return chunks

    def _format_text_chunks(self, raw_chunks_list, parent_id, id_prefix, title, content_type, sourcepage, category, file_id=None):
        """
        Internal method: Formats text and table chunks into dictionaries with IDs.
        """
        formatted_chunks = []
        for index, chunk_text in enumerate(raw_chunks_list, start=1):
            chunk_dict = {
                "id": f"{id_prefix}_{index}",
                "parent_id": parent_id,
                "file_id": file_id,
                "title": title,
                "chunk_type": content_type,
                "sourcepage": sourcepage,
                "category": category,
                "content": chunk_text
            }
                
            formatted_chunks.append(chunk_dict)
        return formatted_chunks

    def _format_image_chunks(self, image_list, parent_id, id_prefix, title, content_type, sourcepage, category, file_id=None):
        """
        Internal method: Formats image chunks with IDs and Title.
        """
        formatted_chunks = []
        for index, img_obj in enumerate(image_list, start=1):
            formatted_chunks.append({
                "id": f"{id_prefix}_{index}",
                "parent_id": parent_id,
                "file_id": file_id,
                "title": title,
                "chunk_type": content_type,
                "sourcepage": sourcepage,
                "category": category,
                "content": img_obj.get("image_description", ""),
                "width": img_obj.get("width"),
                "height": img_obj.get("height"),
            })
        return formatted_chunks

    def process_data(self, data, file_id=None):
        """
        Process a list of dictionaries (the loaded JSON data). 
        Returns a single FLAT list of all chunk dictionaries.
        Ensures unique IDs for multiple elements on the same page.
        """
        all_chunks = []
        
        # Dictionary to track how many elements exist per page 
        # Key: "Title_PageNo", Value: integer counter
        page_counters = {}

        for entry in data:
            try:
                # Generate Common Parent ID if missing
                if "element_id" not in entry:
                    entry["element_id"] = str(uuid.uuid4())
                
                parent_id = entry["element_id"]
                content_type = entry.get("content_type")
                category = entry.get("category") or None
                content = entry.get("page_content")
                sourcepage = entry.get("sourcepage") or None
                
                # Get Title and Page
                raw_title = entry.get("title") or "doc"
                clean_title_str = str(raw_title).replace(".", "_")
                clean_title = "".join(x for x in clean_title_str if x.isalnum() or x in ['_', '-'])
                page_no = entry.get("page_no") or "0"
                
                # --- UNIQUE ID LOGIC ---
                page_key = f"{clean_title}_{page_no}"
                element_index = page_counters.get(page_key, 1)
                page_counters[page_key] = element_index + 1
                id_prefix = f"{clean_title}_{page_no}_{element_index}"
                # --- END UNIQUE ID LOGIC ---

                current_chunks = []

                # --- CASE 1: TEXT ---
                if content_type == "text" and isinstance(content, str):
                    if not content.strip():
                        print(f"⚠ Skipping empty text content for {raw_title} page {page_no}")
                        continue
                    
                    raw_text_chunks = self.text_splitter.split_text(content)
                    current_chunks = self._format_text_chunks(
                        raw_text_chunks, parent_id, id_prefix, raw_title, content_type, sourcepage, category, file_id
                    )
                    print(f"✓ Chunked text from {raw_title} page {page_no}: {len(current_chunks)} chunks")
                
                # --- CASE 2: TABLE ---
                elif content_type == "table" and isinstance(content, str):
                    if not content.strip():
                        print(f"⚠ Skipping empty table content for {raw_title} page {page_no}")
                        continue
                    
                    raw_table_chunks = self._chunk_markdown_table(content)
                    current_chunks = self._format_text_chunks(
                        raw_table_chunks, parent_id, id_prefix, raw_title, content_type, sourcepage, category, file_id
                    )
                    print(f"✓ Chunked table from {raw_title} page {page_no}: {len(current_chunks)} chunks")

                # --- CASE 3: IMAGE ---
                elif content_type == "image" and isinstance(content, list):
                    if not content:
                        print(f"⚠ Skipping empty image list for {raw_title} page {page_no}")
                        continue
                    
                    current_chunks = self._format_image_chunks(
                        content, parent_id, id_prefix, raw_title, content_type, sourcepage, category, file_id
                    )
                    print(f"✓ Processed images from {raw_title} page {page_no}: {len(current_chunks)} images")
                
                else:
                    print(f"⚠ Unrecognized content type: {content_type} for {raw_title} page {page_no}")
                
                if current_chunks:
                    all_chunks.extend(current_chunks)

            except Exception as e:
                print(f"✗ Error processing entry {entry.get('title', 'unknown')}: {str(e)}")
                continue

        # --- ADD NEIGHBOUR IDS ---
        print(f"\nAdding neighbour relationships to {len(all_chunks)} chunks...")
        total_chunks = len(all_chunks)
        for i in range(total_chunks):
            neighbors = []
            
            # Check for previous chunk (if not the first item)
            if i > 0:
                neighbors.append(all_chunks[i-1]["id"])
            
            # Check for next chunk (if not the last item)
            if i < total_chunks - 1:
                neighbors.append(all_chunks[i+1]["id"])
            
            # Assign the list to the current chunk
            all_chunks[i]["neighbour"] = neighbors
        
        print(f"✓ Processing complete: {len(all_chunks)} total chunks created")
        return all_chunks

    def process_file(self, input_file_path, output_file_path):
        """
        Handles File I/O: Reads JSON, processes it, and writes the result.
        """
        try:
            if not os.path.exists(input_file_path):
                raise FileNotFoundError(f"Input file not found at {input_file_path}")

            print(f"Reading from: {input_file_path}")
            with open(input_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Loaded {len(data)} entries from JSON")

            # Process the data using the class logic
            processed_data = self.process_data(data)

            # Write to Output JSON
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            
            print(f"✓ Success! Processed data saved to: {output_file_path}")
            return processed_data

        except FileNotFoundError as e:
            print(f"✗ File error: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {str(e)}")
            raise
        except Exception as e:
            print(f"✗ An error occurred: {str(e)}")
            raise

# ==========================================
# Execution / Usage Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Define Paths
    INPUT_PATH = "/Users/santusahoo/Documents/DAGENT/merged_output.json"
    OUTPUT_PATH = "output_processed.json"

    # 2. Instantiate the class (You can adjust settings here)
    processor = DocumentChunker()

    # 3. Run the processing
    try:
        processor.process_file(INPUT_PATH, OUTPUT_PATH)
    except Exception as e:
        print(f"Processing failed: {e}")
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from typing import List, Dict, Any
import json
from datetime import datetime
from langchain_core.documents import Document
from .chunker import DocumentChunker
from .chroma_manager import ChromaDBManager
from .qa_system import QASystem
from .document_processor import PDFParser

app = FastAPI(
    title="RAG Document Processing API",
    description="Upload, process, and query documents using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
UPLOAD_DIRECTORY = "uploaded_files"
PROCESSED_DIRECTORY = "processed_chunks"
CHROMA_PERSIST_DIRECTORY = "/Users/santusahoo/Documents/DAGENT/rags_test/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_LLM_MODEL = "gemma3:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Initialize Managers ---
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(PROCESSED_DIRECTORY):
    os.makedirs(PROCESSED_DIRECTORY)

db_manager = ChromaDBManager(
    persist_directory=CHROMA_PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_model_name=OLLAMA_EMBEDDING_MODEL
)

chunker = DocumentChunker()

qa_system = QASystem(
    db_path=CHROMA_PERSIST_DIRECTORY,
    collection_name=COLLECTION_NAME,
    embedding_model=OLLAMA_EMBEDDING_MODEL,
    llm_model=OLLAMA_LLM_MODEL
)

# --- Processing Status Tracking ---
processing_status = {}

def get_file_path(file_id: str) -> str:
    """Gets the file path for a given file_id."""
    for filename in os.listdir(UPLOAD_DIRECTORY):
        if filename.startswith(file_id):
            return os.path.join(UPLOAD_DIRECTORY, filename)
    return None

def update_status(file_id: str, step: str, status: str, details: str = ""):
    """Update processing status for a file."""
    if file_id not in processing_status:
        processing_status[file_id] = {
            "file_id": file_id,
            "started_at": datetime.now().isoformat(),
            "steps": {}
        }
    
    processing_status[file_id]["steps"][step] = {
        "status": status,  # "pending", "in_progress", "completed", "failed"
        "details": details,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "vector_store": f"Connected ({db_manager.collection.count()} documents)"
        },
        status_code=200
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file and saves it to the server.
    
    Returns:
        - file_id: Unique identifier for the uploaded file
        - file_name: Original filename
        - message: Confirmation message
    """
    try:
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Initialize status tracking
        update_status(file_id, "upload", "completed", f"Uploaded {file.filename}")
        
        return JSONResponse(
            content={
                "message": "File uploaded successfully",
                "file_id": file_id,
                "file_name": file.filename,
                "next_step": f"POST /process/{file_id}"
            },
            status_code=201
        )
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/process/{file_id}")
async def process_file(file_id: str, skip_image_descriptions: bool = False):
    """
    Processes an uploaded file through the complete pipeline:
    1. Parse/Extract (PDF or JSON)
    2. Chunk the extracted content
    3. Add to vector database
    
    Args:
        file_id: The file ID from upload
        skip_image_descriptions: If True, skips image description generation (faster)
    
    Returns:
        Processing results with statistics and intermediate file paths
    """
    file_path = get_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # --- STEP 1: PARSE/EXTRACT ---
        update_status(file_id, "parse", "in_progress", "Extracting content from file...")
        print(f"\n{'='*60}")
        print(f"PROCESSING FILE: {file_id}")
        print(f"{'='*60}")
        
        parsed_json_path = os.path.join(PROCESSED_DIRECTORY, f"{file_id}_parsed.json")
        
        if file_path.endswith(".pdf"):
            print("▶ Step 1: Parsing PDF...")
            parser = PDFParser(ollama_url=OLLAMA_URL)
            data = parser.parse_pdf(
                file_path,
                output_file=parsed_json_path,
                merge_blocks=True,
                skip_image_descriptions=skip_image_descriptions
            )
        else:
            print("▶ Step 1: Loading JSON...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Still save intermediate JSON
            with open(parsed_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        
        update_status(file_id, "parse", "completed", f"Extracted {len(data)} content blocks")
        print(f"✓ Parsed content saved to: {parsed_json_path}\n")

        # --- STEP 2: CHUNK ---
        update_status(file_id, "chunk", "in_progress", "Chunking extracted content...")
        print("▶ Step 2: Chunking content...")
        
        chunks = chunker.process_data(data, file_id=file_id)
        
        if not chunks:
            raise ValueError("No chunks were generated from the data")
        
        # Save chunks for reference
        chunks_json_path = os.path.join(PROCESSED_DIRECTORY, f"{file_id}_chunks.json")
        with open(chunks_json_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        
        update_status(file_id, "chunk", "completed", f"Created {len(chunks)} chunks")
        print(f"✓ Chunks saved to: {chunks_json_path}\n")

        # --- STEP 3: EMBEDDING & DATABASE ---
        update_status(file_id, "embed", "in_progress", "Adding chunks to vector database...")
        print("▶ Step 3: Adding to vector database...")
        
        documents = []
        for chunk in chunks:
            page_content = chunk.get("content", "")
            metadata = {k: v for k, v in chunk.items() if k != "content"}
            
            # Convert neighbour list to string for storage
            if 'neighbour' in metadata and metadata['neighbour'] is not None:
                metadata['neighbour'] = ", ".join(metadata['neighbour'])
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        # Add all documents to the database
        db_manager.add_documents(documents)
        
        update_status(file_id, "embed", "completed", f"Added {len(documents)} documents to vector store")
        print(f"✓ Added {len(documents)} documents to vector database\n")

        # --- SUCCESS RESPONSE ---
        processing_status[file_id]["completed_at"] = datetime.now().isoformat()
        processing_status[file_id]["success"] = True
        
        response = {
            "message": "File processed successfully",
            "file_id": file_id,
            "file_name": os.path.basename(file_path),
            "statistics": {
                "content_blocks": len(data),
                "chunks_created": len(chunks),
                "documents_indexed": len(documents),
                "total_vector_store_docs": db_manager.collection.count()
            },
            "intermediate_files": {
                "parsed_json": parsed_json_path,
                "chunks_json": chunks_json_path
            },
            "next_step": f"GET /qa/?query=your_question",
            "status": processing_status.get(file_id, {})
        }
        
        print(f"{'='*60}")
        print(f"✓ PROCESSING COMPLETE")
        print(f"{'='*60}\n")
        
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Processing failed: {error_msg}\n")
        
        # Determine which step failed
        current_status = processing_status.get(file_id, {}).get("steps", {})
        failed_step = None
        
        if "parse" not in current_status or current_status["parse"]["status"] != "completed":
            failed_step = "parse"
        elif "chunk" not in current_status or current_status["chunk"]["status"] != "completed":
            failed_step = "chunk"
        else:
            failed_step = "embed"
        
        update_status(file_id, failed_step, "failed", error_msg)
        processing_status[file_id]["success"] = False
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": error_msg,
                "failed_step": failed_step,
                "status": processing_status.get(file_id, {})
            }
        )

@app.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """
    Get the processing status of a file.
    
    Returns:
        Processing steps with timestamps and statuses
    """
    if file_id not in processing_status:
        raise HTTPException(status_code=404, detail="No status found for this file_id")
    
    return JSONResponse(content=processing_status[file_id], status_code=200)

@app.get("/files/")
async def get_files() -> List[Dict[str, Any]]:
    """
    Lists all uploaded files with their processing status.
    
    Returns:
        List of files with file_id, file_name, and processing status
    """
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        file_id = filename.split("_")[0]
        file_name = "_".join(filename.split("_")[1:])
        status = processing_status.get(file_id, {}).get("success", None)
        
        files.append({
            "file_id": file_id,
            "file_name": file_name,
            "processed": status
        })
    return files

@app.get("/files/{file_id}/chunks/")
async def get_chunks(file_id: str):
    """
    Retrieves all chunks for a specific file from the vector database.
    """
    try:
        results = db_manager.collection.get(where={"file_id": file_id})
        return JSONResponse(
            content={
                "file_id": file_id,
                "chunk_count": len(results.get("ids", [])),
                "chunks": results
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")

@app.get("/search/")
async def search(query: str, n_results: int = 5):
    """
    Performs a similarity search in the vector store.
    
    Args:
        query: Search query string
        n_results: Number of results to return (default: 5)
    
    Returns:
        List of similar documents with metadata
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    try:
        print(f"\n🔍 Search query: {query}")
        results = db_manager.search(query_text=query, n_results=n_results)
        
        # Format results for better readability
        formatted_results = {
            "query": query,
            "num_results": 0,
            "results": []
        }
        
        if not results:
            return JSONResponse(content=formatted_results, status_code=200)
        
        # Chroma returns nested lists: {"ids": [[...]], "documents": [[...]], "metadatas": [[...]]}
        ids = results.get("ids", [[]])[0] if results.get("ids") else []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        
        formatted_results["num_results"] = len(ids)
        
        # Zip results together
        for idx, (doc_id, content, metadata) in enumerate(zip(ids, documents, metadatas), 1):
            distance = distances[idx - 1] if idx - 1 < len(distances) else None
            
            formatted_results["results"].append({
                "rank": idx,
                "id": doc_id,
                "distance": round(distance, 4) if distance else None,
                "title": metadata.get("title", "Unknown") if isinstance(metadata, dict) else "Unknown",
                "sourcepage": metadata.get("sourcepage", "N/A") if isinstance(metadata, dict) else "N/A",
                "content_preview": content[:300] + "..." if len(content) > 300 else content
            })
        
        return JSONResponse(content=formatted_results, status_code=200)
    except Exception as e:
        import traceback
        print(f"Search error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/qa/")
async def qa(query: str):
    """
    Asks a question to the RAG QA system.
    The system retrieves relevant documents and generates an answer.
    
    Args:
        query: The question to ask
    
    Returns:
        The answer based on retrieved documents
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    try:
        print(f"\n❓ QA Query: {query}")
        answer = qa_system.ask_question(query)
        
        return JSONResponse(
            content={
                "query": query,
                "answer": answer
            },
            status_code=200
        )
    except Exception as e:
        print(f"✗ QA Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """
    Deletes a file and all its associated chunks from the database.
    """
    file_path = get_file_path(file_id)
    if not file_path:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Delete the file from the upload directory
        os.remove(file_path)

        # Delete intermediate files
        parsed_json = os.path.join(PROCESSED_DIRECTORY, f"{file_id}_parsed.json")
        chunks_json = os.path.join(PROCESSED_DIRECTORY, f"{file_id}_chunks.json")
        
        if os.path.exists(parsed_json):
            os.remove(parsed_json)
        if os.path.exists(chunks_json):
            os.remove(chunks_json)

        # Delete the associated chunks from the database
        db_manager.collection.delete(where={"file_id": file_id})
        
        # Clear status
        if file_id in processing_status:
            del processing_status[file_id]

        return JSONResponse(
            content={
                "message": f"File {file_id} and its associated chunks have been deleted.",
                "total_remaining_docs": db_manager.collection.count()
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.delete("/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """
    Deletes a specific chunk from the database.
    """
    try:
        db_manager.delete_chunk_by_id(chunk_id)
        return JSONResponse(
            content={
                "message": f"Chunk {chunk_id} has been deleted.",
                "total_remaining_docs": db_manager.collection.count()
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {str(e)}")

@app.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """
    Retrieves a specific chunk by its ID.
    """
    try:
        chunk = db_manager.get_chunk_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        return JSONResponse(content=chunk, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunk: {str(e)}")

@app.delete("/collection")
async def clear_collection():
    """
    Clears the entire collection.
    ⚠️ Warning: This action is irreversible!
    """
    try:
        db_manager.clear_collection()
        processing_status.clear()
        return JSONResponse(
            content={
                "message": "Collection has been cleared.",
                "documents_remaining": 0
            },
            status_code=200
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
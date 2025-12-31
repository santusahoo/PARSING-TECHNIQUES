
import json
import os
from typing import List, Optional
from langchain_core.documents import Document
from pydantic import BaseModel, ValidationError
from rags_test.chroma_manager import ChromaDBManager

# --- Configuration ---
INPUT_JSON_PATH = "/Users/santusahoo/Documents/DAGENT/output_processed.json"
COLLECTION_NAME = "document_chunks_collection"
CHROMA_PERSIST_DIRECTORY = "/Users/santusahoo/Documents/DAGENT/rags_test/chroma_db"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" # A good default for Ollama embeddings


class Chunk(BaseModel):
    """Pydantic model to validate the structure of each chunk."""
    id: str
    parent_id: str
    title: str
    chunk_type: str
    sourcepage: str
    category: str
    content: str
    neighbour: Optional[List[str]] = None


class EmbeddingGenerator:
    """
    Handles loading, validating, and generating embeddings for document chunks.
    """
    def __init__(self, input_path: str, db_path: str, collection_name: str, model_name: str):
        self.input_path = input_path
        # Initialize the ChromaDBManager to handle all DB interactions
        self.db_manager = ChromaDBManager(
            persist_directory=db_path,
            collection_name=collection_name,
            embedding_model_name=model_name
        )
        print(f"EmbeddingGenerator initialized, using ChromaDBManager for collection '{collection_name}'.")

    def load_and_validate_chunks(self) -> List[Chunk]:
        """Loads chunks from a JSON file and validates them using the Pydantic model."""
        if not os.path.exists(self.input_path):
            print(f"Error: Input file not found at {self.input_path}")
            return []

        print(f"Loading and validating chunks from {self.input_path}...")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        try:
            validated_chunks = [Chunk.model_validate(item) for item in data]
            print(f"Successfully validated {len(validated_chunks)} chunks.")
            return validated_chunks
        except ValidationError as e:
            print(f"Data validation error: {e}")
            return []

    def add_chunks_to_db(self, chunks: List[Chunk]):
        """Converts chunks to LangChain Documents and adds them to ChromaDB."""
        if not chunks:
            print("No chunks to process.")
            return

        # Convert Pydantic models to LangChain Document objects
        documents = []
        for chunk in chunks:
            # The main text content for embedding
            page_content = chunk.content
            # All other fields become metadata
            metadata = chunk.model_dump(exclude={'content'})

            # --- FIX: Convert list to a ChromaDB-compatible type (string) ---
            # ChromaDB metadata values cannot be lists. We'll join the list into a single string.
            if 'neighbour' in metadata and metadata['neighbour'] is not None:
                metadata['neighbour'] = ", ".join(metadata['neighbour'])

            documents.append(Document(page_content=page_content, metadata=metadata))

        # Use the manager to add the documents. It handles embedding and storage.
        self.db_manager.add_documents(documents)
        print(f"\n✅ Pipeline step complete: {len(documents)} documents added to the vector store.")

    def run(self):
        """Executes the full embedding generation pipeline."""
        validated_chunks = self.load_and_validate_chunks()
        if validated_chunks:
            # 🔥 Clear existing collection before inserting new data
            print("\n🧹 Clearing existing ChromaDB collection before re-embedding...")
            self.db_manager.clear_collection()

            # Now add new chunks
            self.add_chunks_to_db(validated_chunks)



if __name__ == "__main__":
    print("--- Running Embedding Generator Standalone ---")
    # Ensure you have Ollama running with the specified model, e.g.,
    # `ollama run nomic-embed-text`
    generator = EmbeddingGenerator(
        input_path=INPUT_JSON_PATH,
        db_path=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        model_name=OLLAMA_EMBEDDING_MODEL
    )
    generator.run()
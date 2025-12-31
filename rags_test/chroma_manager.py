import chromadb
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import hashlib

class ChromaDBManager:
    """
    Enhanced ChromaDB manager with proper embedding verification and debugging.
    """
    def __init__(self, persist_directory: str, collection_name: str, embedding_model_name: str):
        """
        Initializes the ChromaDB client and the specific collection with embeddings.

        Args:
            persist_directory (str): The file path where the database will be stored.
            collection_name (str): The name of the collection to manage.
            embedding_model_name (str): The name of the Ollama model to use for embeddings.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Initialize a persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize the embedding function
        self.embedding_function = OllamaEmbeddings(model=self.embedding_model_name)
        
        # Verify Ollama connection and get actual model name
        print(f"✅ Testing Ollama embeddings with model '{self.embedding_model_name}'...")
        try:
            test_embedding = self.embedding_function.embed_query("test")
            print(f"✅ Ollama connection successful! Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            # Try with :latest suffix
            print(f"⚠️  First attempt failed, trying with :latest suffix...")
            try:
                model_with_latest = f"{self.embedding_model_name}:latest"
                self.embedding_function = OllamaEmbeddings(model=model_with_latest)
                test_embedding = self.embedding_function.embed_query("test")
                self.embedding_model_name = model_with_latest
                print(f"✅ Ollama connection successful with '{model_with_latest}'!")
                print(f"   Embedding dimension: {len(test_embedding)}")
            except Exception as e2:
                print(f"❌ ERROR: Cannot connect to Ollama embeddings: {e2}")
                print(f"   Make sure Ollama is running: ollama serve")
                print(f"   And model is available: ollama pull {self.embedding_model_name}")
                raise
        
        # Get or create the collection with embeddings
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance for embeddings
        )
        print(f"✅ ChromaDBManager initialized for collection '{self.collection_name}'")
        print(f"   Location: {self.persist_directory}")
        print(f"   Current document count: {self.collection.count()}")

    def _generate_deterministic_id(self, content: str) -> str:
        """
        Generates a deterministic SHA-256 hash for a document's content to use as a unique ID.
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def add_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Adds a list of LangChain Document objects to the collection with embeddings.
        
        Returns:
            Dictionary with statistics about the operation
        """
        if not documents:
            print("⚠️  No documents provided to add.")
            return {"status": "error", "message": "No documents provided"}

        try:
            contents = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [self._generate_deterministic_id(content) for content in contents]
            
            print(f"📦 Processing {len(documents)} documents...")
            print(f"   Generating embeddings with '{self.embedding_model_name}'...")
            
            # Generate embeddings
            embeddings = self.embedding_function.embed_documents(contents)
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            print(f"   Embedding dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            
            # Add to collection
            print(f"📥 Adding documents to ChromaDB...")
            self.collection.add(
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            total_count = self.collection.count()
            print(f"✅ Successfully added {len(documents)} documents")
            print(f"   Total documents in collection: {total_count}")
            
            return {
                "status": "success",
                "documents_added": len(documents),
                "total_documents": total_count,
                "embedding_dimension": len(embeddings[0]) if embeddings else None
            }

        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            raise

    def search(self, query_text: str, n_results: int = 5) -> Optional[Dict[str, Any]]:
        """
        Performs a similarity search in the collection using embeddings.
        """
        if not query_text:
            print("⚠️  Query text cannot be empty.")
            return None
            
        try:
            print(f"\n🔍 Searching for: '{query_text}'...")
            
            # Generate embedding for the query
            query_embedding = self.embedding_function.embed_query(query_text)
            print(f"✅ Query embedding generated (dimension: {len(query_embedding)})")
            
            # Perform the query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"✅ Found {len(results.get('ids', [[]])[0])} results")
            
            return results

        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            raise

    def verify_embeddings(self) -> Dict[str, Any]:
        """
        Verifies that embeddings are properly stored in the collection.
        
        Returns:
            Verification results with detailed information
        """
        try:
            print("\n🔍 Verifying embeddings in collection...")
            
            # Get a sample of documents
            sample_size = min(5, self.collection.count())
            all_items = self.collection.get(limit=sample_size)
            
            total_docs = self.collection.count()
            
            verification_result = {
                "total_documents": total_docs,
                "sample_size": sample_size,
                "documents_have_embeddings": total_docs > 0,
                "documents_with_metadata": len(all_items.get('metadatas', [])),
                "sample_documents": []
            }
            
            # Check each sample document
            for i, (doc_id, content, metadata) in enumerate(zip(
                all_items.get('ids', []),
                all_items.get('documents', []),
                all_items.get('metadatas', [])
            )):
                verification_result["sample_documents"].append({
                    "index": i + 1,
                    "id": doc_id,
                    "has_content": bool(content),
                    "content_length": len(content) if content else 0,
                    "has_metadata": bool(metadata),
                    "metadata_keys": list(metadata.keys()) if metadata else []
                })
            
            print(f"✅ Verification complete:")
            print(f"   Total documents: {total_docs}")
            print(f"   Documents with metadata: {verification_result['documents_with_metadata']}")
            print(f"   Sample documents verified: {len(verification_result['sample_documents'])}")
            
            return verification_result

        except Exception as e:
            print(f"❌ Verification error: {str(e)}")
            raise

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chunk by its unique ID."""
        try:
            result = self.collection.get(ids=[chunk_id], include=["metadatas", "documents"])
            if result and result['ids']:
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0]
                }
            return None
        except Exception as e:
            print(f"❌ Error getting chunk: {str(e)}")
            return None

    def delete_chunk_by_id(self, chunk_id: str):
        """Deletes a single chunk by its unique ID."""
        try:
            self.collection.delete(ids=[chunk_id])
            print(f"✅ Chunk '{chunk_id}' deleted")
        except Exception as e:
            print(f"❌ Error deleting chunk: {str(e)}")
            raise

    def get_all_chunks(self, limit: int = 100) -> Dict[str, Any]:
        """Retrieves all chunks from the collection."""
        try:
            return self.collection.get(limit=limit, include=["metadatas", "documents"])
        except Exception as e:
            print(f"❌ Error getting chunks: {str(e)}")
            raise

    def clear_collection(self):
        """Clears all items within the collection."""
        try:
            print(f"🧹 Clearing collection '{self.collection_name}'...")
            all_items = self.collection.get()
            if all_items['ids']:
                self.collection.delete(ids=all_items['ids'])
                print("✅ Collection cleared")
            else:
                print("ℹ️  Collection was already empty")
        except Exception as e:
            print(f"❌ Error clearing collection: {str(e)}")
            raise

    def delete_collection(self):
        """Deletes the entire collection from the database."""
        try:
            print(f"💥 Deleting entire collection '{self.collection_name}'...")
            self.client.delete_collection(name=self.collection_name)
            print("✅ Collection deleted")
        except Exception as e:
            print(f"❌ Error deleting collection: {str(e)}")
            raise

    def count(self) -> int:
        """Returns the total number of items in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"❌ Error getting count: {str(e)}")
            return 0



# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = "/Users/santusahoo/Documents/DAGENT/rags_test/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


# ==========================================
# Test/Verification Script
# ==========================================
if __name__ == "__main__":
    
    # 1. Initialize manager
    print("=== Initializing ChromaDB Manager ===\n")
    db_manager = ChromaDBManager(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        embedding_model_name=OLLAMA_EMBEDDING_MODEL
    )

    # 2. Verify embeddings
    print("\n=== Verifying Embeddings ===")
    verification = db_manager.verify_embeddings()
    
    if verification['total_documents'] > 0:
        print("✅ Embeddings are properly stored!")
        print(f"\n📊 Summary:")
        print(f"   • Total documents: {verification['total_documents']}")
        print(f"   • Documents verified: {verification['sample_size']}")
        
        # 3. Test search
        print("\n=== Testing Search ===")
        search_results = db_manager.search(
            query_text="Cortex-M3 applications",
            n_results=3
        )
        
        if search_results and search_results['ids']:
            print(f"\n✅ Search working! Found {len(search_results['ids'][0])} results")
            for i, (doc_id, content) in enumerate(zip(
                search_results['ids'][0],
                search_results['documents'][0]
            ), 1):
                print(f"\n   Result {i}:")
                print(f"   ID: {doc_id[:16]}...")
                print(f"   Content: {content[:100]}...")
    else:
        print("❌ No documents found in collection!")
        print("   Please ensure documents have been added first.")
# --- Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List

# --- Configuration ---
CHROMA_PERSIST_DIRECTORY = "/Users/santusahoo/Documents/DAGENT/rags_test/chroma_db"
COLLECTION_NAME = "document_chunks_collection"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "gemma3:latest"


class QASystem:
    """
    Encapsulates the logic for a Retrieval-Augmented Generation (RAG) system.
    It loads a vector store, sets up a retriever, and creates a question-answering chain.
    """

    def __init__(self, db_path, collection_name, embedding_model, llm_model):
        print("--- Initializing RAG QA System ---")

        # 1. Initialize Embeddings
        embedding_function = OllamaEmbeddings(model=embedding_model)

        # 2. Load the Chroma Vector Store
        self.vector_store = Chroma(
            persist_directory=db_path,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        print(f"✅ Vector store loaded from '{db_path}' with {self.vector_store._collection.count()} documents.")

        # 3. Create a Retriever with optimized search parameters
        # Increased k to 5 to provide more context to the LLM
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Retriever created.")

        # 4. Define the RAG Prompt Template
        rag_template = """You are an expert assistant. Answer the user's question based ONLY on the following context.
Provide a concise, direct, and informative answer. If the answer is found in the context, cite the source.
If the information is not in the context, say "I could not find this information in the provided documents."

<context>
{context}
</context>

Question: {question}
Answer:"""
        rag_prompt = ChatPromptTemplate.from_template(rag_template)

        # 5. Initialize the LLM
        llm = ChatOllama(model=llm_model, temperature=0.2)
        print(f"✅ LLM model '{llm_model}' initialized.")

        # 6. Format retrieved documents with metadata
        def format_docs(docs: List) -> str:
            """Format retrieved documents with their metadata."""
            if not docs:
                return "No relevant documents found."
            
            formatted = []
            for idx, doc in enumerate(docs, 1):
                title = doc.metadata.get('title', 'Unknown')
                sourcepage = doc.metadata.get('sourcepage', 'N/A')
                content = doc.page_content.strip()
                
                formatted.append(
                    f"[Document {idx}]\n"
                    f"Source: {title}\n"
                    f"Page: {sourcepage}\n"
                    f"Content: {content}"
                )
            
            return "\n\n".join(formatted)

        # 7. Create the RAG Chain
        # FIXED: Simplified chain without the problematic retrieval_gate
        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        print("✅ RAG chain created and ready.")

    def ask_question(self, question: str) -> str:
        """
        Asks a question to the RAG chain and returns the answer.
        
        Args:
            question: The user's question
            
        Returns:
            The answer from the RAG system
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        print(f"\n{'='*70}")
        print(f"🔎 Question: '{question}'")
        print(f"{'='*70}")
        
        try:
            # Invoke the chain
            answer = self.rag_chain.invoke(question)
            
            print(f"💡 Answer:\n{answer}")
            print(f"{'='*70}\n")
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

    def search_similar(self, query: str, k: int = 5) -> List[dict]:
        """
        Search for similar documents without asking a full question.
        Useful for debugging retrieval.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        docs = self.retriever.invoke(query)
        
        results = []
        for idx, doc in enumerate(docs, 1):
            result = {
                "rank": idx,
                "title": doc.metadata.get('title', 'Unknown'),
                "sourcepage": doc.metadata.get('sourcepage', 'N/A'),
                "chunk_id": doc.metadata.get('id', 'N/A'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            results.append(result)
            print(f"\n[Result {idx}] {result['title']}")
            print(f"   Source: {result['sourcepage']}")
            print(f"   Preview: {result['content_preview']}")
        
        return results


if __name__ == "__main__":
    qa_system = QASystem(
        CHROMA_PERSIST_DIRECTORY,
        COLLECTION_NAME,
        OLLAMA_EMBEDDING_MODEL,
        OLLAMA_LLM_MODEL
    )
    
    # Test with the original question
    question = "total tax amount"
    
    print("\n" + "="*70)
    print("TEST 1: Document Retrieval")
    print("="*70)
    qa_system.search_similar(question, k=10)
    
    print("\n" + "="*70)
    print("TEST 2: Full RAG Question-Answer")
    print("="*70)
    qa_system.ask_question(question)
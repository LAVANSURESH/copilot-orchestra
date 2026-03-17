## Plan: Enterprise Chatbot Framework

A centralized chatbot framework for enterprise environments with document access, semantic search capabilities, and integration with internal knowledge bases (codebase and Confluence). The framework will provide a modular architecture supporting document ingestion, vector embeddings, semantic search, and conversational AI interfaces.

**Phases: 8 phases**

1. **Phase 1: Core Framework Structure & Configuration**
    - **Objective:** Establish the foundational project structure with proper configuration management for enterprise settings (API keys, database connections, document sources)
    - **Files/Functions to Modify/Create:**
        - `src/config/settings.py` - Configuration management with environment variables
        - `src/config/__init__.py` - Package initialization
        - `pyproject.toml` or `setup.py` - Python project configuration
        - `requirements.txt` - Python dependencies
        - `.env.example` - Example environment configuration
        - `README_FRAMEWORK.md` - Framework documentation
    - **Tests to Write:**
        - `tests/test_config.py::test_load_config_from_env`
        - `tests/test_config.py::test_missing_required_config_raises_error`
        - `tests/test_config.py::test_default_config_values`
    - **Steps:**
        1. Write tests for configuration loading and validation
        2. Run tests to see them fail
        3. Create project structure with src/ and tests/ directories
        4. Implement configuration management with pydantic or dataclasses
        5. Run tests to confirm they pass
        6. Lint and format code

2. **Phase 2: Document Ingestion Module**
    - **Objective:** Create a modular document ingestion system that can load documents from various sources (local files, APIs) and parse different formats (PDF, Markdown, Confluence)
    - **Files/Functions to Modify/Create:**
        - `src/ingestion/base.py` - Abstract base class `DocumentSource` with `load()` method
        - `src/ingestion/file_source.py` - `FileSystemSource` class for local files
        - `src/ingestion/confluence_source.py` - `ConfluenceSource` class for Confluence API
        - `src/ingestion/parsers.py` - Document parsing utilities for PDF, MD, etc.
        - `src/models/document.py` - `Document` dataclass (id, content, metadata, source)
    - **Tests to Write:**
        - `tests/ingestion/test_file_source.py::test_load_markdown_files`
        - `tests/ingestion/test_file_source.py::test_load_pdf_files`
        - `tests/ingestion/test_confluence_source.py::test_fetch_confluence_pages`
        - `tests/ingestion/test_parsers.py::test_parse_markdown`
        - `tests/ingestion/test_parsers.py::test_parse_pdf`
    - **Steps:**
        1. Write tests for document loading from file system
        2. Run tests to see them fail
        3. Implement Document model and base DocumentSource interface
        4. Implement FileSystemSource with support for markdown and text files
        5. Run tests to confirm file source tests pass
        6. Write tests for Confluence integration
        7. Run tests to see Confluence tests fail
        8. Implement ConfluenceSource with API integration
        9. Run all tests to confirm they pass
        10. Lint and format code

3. **Phase 3: Code Repository Analysis Module**
    - **Objective:** Build functionality to analyze code repositories, extract code structure (functions, classes, modules), and create searchable metadata about the codebase
    - **Files/Functions to Modify/Create:**
        - `src/ingestion/code_source.py` - `CodeRepositorySource` class
        - `src/analysis/code_analyzer.py` - Code parsing using AST (Python) or tree-sitter
        - `src/models/code_entity.py` - `CodeEntity` dataclass (file, function, class, docstring)
    - **Tests to Write:**
        - `tests/ingestion/test_code_source.py::test_load_python_files_from_repo`
        - `tests/analysis/test_code_analyzer.py::test_extract_functions_from_python`
        - `tests/analysis/test_code_analyzer.py::test_extract_classes_from_python`
        - `tests/analysis/test_code_analyzer.py::test_extract_docstrings`
    - **Steps:**
        1. Write tests for loading code files from repository
        2. Run tests to see them fail
        3. Implement CodeEntity model
        4. Implement CodeRepositorySource to traverse and load code files
        5. Run tests to confirm loading tests pass
        6. Write tests for code analysis and AST parsing
        7. Run tests to see analysis tests fail
        8. Implement CodeAnalyzer using Python AST module
        9. Run all tests to confirm they pass
        10. Lint and format code

4. **Phase 4: Vector Embedding & Storage**
    - **Objective:** Implement vector embedding generation for documents and code, with a vector database for similarity search (using sentence-transformers and ChromaDB/FAISS)
    - **Files/Functions to Modify/Create:**
        - `src/embeddings/embedder.py` - `EmbeddingModel` class using sentence-transformers
        - `src/storage/vector_store.py` - `VectorStore` interface and implementation with ChromaDB
        - `src/storage/__init__.py` - Package initialization
    - **Tests to Write:**
        - `tests/embeddings/test_embedder.py::test_generate_embedding_for_text`
        - `tests/embeddings/test_embedder.py::test_embedding_dimension_consistency`
        - `tests/storage/test_vector_store.py::test_add_documents_to_store`
        - `tests/storage/test_vector_store.py::test_similarity_search`
        - `tests/storage/test_vector_store.py::test_search_with_metadata_filter`
    - **Steps:**
        1. Write tests for embedding generation
        2. Run tests to see them fail
        3. Implement EmbeddingModel with sentence-transformers
        4. Run embedding tests to confirm they pass
        5. Write tests for vector storage and retrieval
        6. Run tests to see storage tests fail
        7. Implement VectorStore with ChromaDB integration
        8. Run all tests to confirm they pass
        9. Lint and format code

5. **Phase 5: Semantic Search & Retrieval**
    - **Objective:** Build a semantic search system that can query the vector store, rank results, and retrieve relevant documents/code snippets based on user queries
    - **Files/Functions to Modify/Create:**
        - `src/retrieval/retriever.py` - `SemanticRetriever` class with query() method
        - `src/retrieval/ranker.py` - Result ranking and re-ranking logic
        - `src/models/search_result.py` - `SearchResult` dataclass
    - **Tests to Write:**
        - `tests/retrieval/test_retriever.py::test_retrieve_documents_by_query`
        - `tests/retrieval/test_retriever.py::test_retrieve_code_entities_by_query`
        - `tests/retrieval/test_retriever.py::test_hybrid_search_documents_and_code`
        - `tests/retrieval/test_ranker.py::test_rerank_results_by_relevance`
    - **Steps:**
        1. Write tests for semantic retrieval of documents
        2. Run tests to see them fail
        3. Implement SemanticRetriever with query embedding and vector search
        4. Run retrieval tests to confirm they pass
        5. Write tests for result ranking
        6. Run tests to see ranking tests fail
        7. Implement ranking logic for relevance scoring
        8. Run all tests to confirm they pass
        9. Lint and format code

6. **Phase 6: Semantic Mapping Between Code & Documentation**
    - **Objective:** Create a system to establish and maintain semantic relationships between code entities and documentation (e.g., linking functions to relevant Confluence pages)
    - **Files/Functions to Modify/Create:**
        - `src/mapping/semantic_mapper.py` - `SemanticMapper` class
        - `src/mapping/link_generator.py` - Functions to generate bidirectional links
        - `src/storage/mapping_store.py` - Store for semantic relationships
    - **Tests to Write:**
        - `tests/mapping/test_semantic_mapper.py::test_find_related_docs_for_code`
        - `tests/mapping/test_semantic_mapper.py::test_find_related_code_for_doc`
        - `tests/mapping/test_link_generator.py::test_generate_bidirectional_links`
        - `tests/storage/test_mapping_store.py::test_store_and_retrieve_mappings`
    - **Steps:**
        1. Write tests for finding related documentation for code
        2. Run tests to see them fail
        3. Implement SemanticMapper to compute similarity between code and docs
        4. Run mapping tests to confirm they pass
        5. Write tests for storing and retrieving semantic mappings
        6. Run tests to see storage tests fail
        7. Implement MappingStore for persisting relationships
        8. Run all tests to confirm they pass
        9. Lint and format code

7. **Phase 7: Chatbot Interface & Conversation Management**
    - **Objective:** Implement the conversational AI interface using LLM integration (OpenAI/Anthropic) with context management and conversation history
    - **Files/Functions to Modify/Create:**
        - `src/chatbot/bot.py` - `ChatBot` class with chat() method
        - `src/chatbot/llm_client.py` - LLM API client (OpenAI/Anthropic)
        - `src/chatbot/context_manager.py` - Conversation context and history management
        - `src/chatbot/prompts.py` - System prompts and prompt templates
    - **Tests to Write:**
        - `tests/chatbot/test_bot.py::test_simple_query_response`
        - `tests/chatbot/test_bot.py::test_query_with_document_context`
        - `tests/chatbot/test_bot.py::test_query_with_code_context`
        - `tests/chatbot/test_context_manager.py::test_maintain_conversation_history`
        - `tests/chatbot/test_llm_client.py::test_call_llm_api`
    - **Steps:**
        1. Write tests for chatbot query handling
        2. Run tests to see them fail
        3. Implement LLM client for API calls (with mocking for tests)
        4. Implement ChatBot with semantic retrieval integration
        5. Run chatbot tests to confirm they pass
        6. Write tests for conversation context management
        7. Run tests to see context tests fail
        8. Implement ContextManager for history tracking
        9. Run all tests to confirm they pass
        10. Lint and format code

8. **Phase 8: CLI Demo & Integration**
    - **Objective:** Create a command-line interface to demonstrate the complete framework functionality: ingest documents/code, build indexes, and interact with the chatbot
    - **Files/Functions to Modify/Create:**
        - `src/cli/main.py` - CLI entry point using argparse or click
        - `src/cli/commands.py` - Command implementations (ingest, query, interactive)
        - `src/pipeline/indexer.py` - Pipeline to orchestrate ingestion and indexing
    - **Tests to Write:**
        - `tests/pipeline/test_indexer.py::test_index_documents_pipeline`
        - `tests/pipeline/test_indexer.py::test_index_code_pipeline`
        - `tests/cli/test_commands.py::test_ingest_command_execution`
        - `tests/cli/test_commands.py::test_query_command_execution`
    - **Steps:**
        1. Write tests for the indexing pipeline
        2. Run tests to see them fail
        3. Implement Indexer to coordinate document/code ingestion and embedding
        4. Run pipeline tests to confirm they pass
        5. Write tests for CLI commands
        6. Run tests to see CLI tests fail
        7. Implement CLI with ingest, query, and interactive commands
        8. Run all tests to confirm they pass
        9. Lint and format code
        10. Create documentation for using the CLI

**Open Questions**

1. **LLM Provider Preference:** Should we default to OpenAI (GPT-4) or Anthropic (Claude)? Or support both with a configuration option?
2. **Vector Database:** ChromaDB (simpler, embedded) vs. Pinecone/Weaviate (production-scale)? Recommend ChromaDB for initial implementation with interface for extensibility.
3. **Code Analysis Scope:** Focus only on Python initially, or support multiple languages (JavaScript, Java) using tree-sitter from the start?
4. **Confluence Authentication:** Support which auth methods - API token, OAuth, or both?
5. **Deployment Target:** Is this meant to run as a standalone service (FastAPI/Flask server) or just as a library/CLI? Initial implementation as library/CLI with optional server in future?

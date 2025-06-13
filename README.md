# TDS-Project-1

## Project Structure

- **v3-app.py**: This script is used for building embeddings, indexing data, and initializing the full Retrieval-Augmented Generation (RAG) system. It processes markdown files and forum data, generates embeddings, and populates the Pinecone index. Use this script for data preparation and system initialization.

- **main.py**: This is the lightweight, deployable API server. It only queries the existing Pinecone index and generates answers using the pre-built embeddings. It is optimized for production deployment and does not perform any heavy data processing or embedding generation at runtime.

## Usage

- Run `v3-app.py` to build and index your data (only needed when updating or initializing the database).
- Deploy `main.py` as your production API for fast, efficient question answering using the pre-built index.

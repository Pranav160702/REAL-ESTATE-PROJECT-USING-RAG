# REAL-ESTATE-PROJECT-USING-RAG

## Overview
The Real Estate Query System is a hybrid search application that combines MongoDB and Retrieval-Augmented Generation (RAG) techniques to provide intelligent and context-aware real estate search capabilities. It allows users to query properties using natural language and fetch results from a MongoDB database while enhancing responses using a vector-based search index.

---

## Features
- **Natural Language Processing**: Extracts user intent and context from natural language queries using OpenAI's GPT-3.5-turbo.
- **Hybrid Search**: Combines MongoDB-based searches with RAG for enhanced results.
- **Dynamic Query Modifications**: Refines previous queries based on new user inputs.
- **Property Management**: Add, store, and index property details with MongoDB and vector-based search.
- **Query History**: Maintains a history of user queries for personalization and relevance.

---

## Technologies Used
- **Programming Language**: Python
- **Database**: MongoDB Atlas
- **Vector Search**: LlamaIndex with MongoDBAtlasVectorSearch
- **AI Model**: OpenAI GPT-3.5-turbo for context extraction
- **Embeddings**: OpenAIEmbedding
- **Environment Management**: Python `dotenv` for managing environment variables

---

## Setup and Installation

### Prerequisites
1. Python 3.8 or higher
2. MongoDB Atlas account
3. OpenAI API Key

### Installation Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/Pranav160702/REAL-ESTATE-PROJECT-USING-RAG.git
   cd REAL-ESTATE-PROJECT-USING-RAG


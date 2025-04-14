# Semantic_Search
# Semantic Movie Recommender Chat

This project is a semantic search engine for movies using the IMDB Top 1000 dataset and an interactive chat interface built with Streamlit. It leverages Sentence Transformers to generate embeddings, stores them in a Pinecone vector database, and retrieves similar movies using cosine similarity. The interface displays the conversation with user messages and bot responses, integrating progress bars using stqdm and a chat message component via st-chat-message.

## Features

- **Text Embeddings:** Uses the Hugging Face model `all-MiniLM-L6-v2` from Sentence Transformers to create rich embeddings from concatenated movie fields (title, genre, and overview).
- **Vector Database:** Stores embeddings with metadata in Pinecone for rapid similarity search.
- **Chat-style UI:** Implements a conversation-style interface using Streamlit and st-chat-message.
- **Progress Visualization:** Uses stqdm to show a progress bar while processing queries.

## Prerequisites

- Python >= 3.8
- Node.js (>= 18.x), Yarn (>= 1.22.x), and Poetry (>= 1.2.x) for the st-chat-message component
- A [Pinecone account](https://www.pinecone.io/) and an API key

## Project Structure
1. *main.py:*  
   - *Data Preparation:* Reads and preprocesses a movie dataset (imdb_top_1000.csv).  
   - *Embedding Generation:* Uses a Sentence Transformer model (all-MiniLM-L12-v2 or all-MiniLM-L12-v2) to generate 384-dimensional embeddings from concatenated movie fields (Series_Title, Genre, Overview).  
   - *Pinecone Indexing:* Creates (if necessary) and uploads embeddings to a Pinecone index named imdb-movies-index in the "default" namespace.

2. *app.py:*  
   - *User Interface:* Provides a simple interface for users to enter a description of the type of movie they want to watch.
   - *Semantic Search:* Uses the same Sentence Transformer model to encode the query and retrieve the top 5 most similar movie entries from the Pinecone index.
   - *Display Results:* Shows each recommended movie's title, genre, overview, and similarity score.


## Code Execution
- create a virtual enviornment and install requirments.txt (pip install -r requirements.txt)
- python main.py in a seperate terminal
- streamlit run app.py in seperate terminal


## 🙌 Acknowledgments

- [Pinecone](https://www.pinecone.io/)
- [Streamlit](https://streamlit.io/)
- [HuggingFace Transformers](https://huggingface.co/sentence-transformers)
- [`st-chat-message`](https://github.com/AI-Yash/st-chat-message)
- [`stqdm`](https://github.com/wznbg/stqdm)


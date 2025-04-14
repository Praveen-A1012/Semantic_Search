import os
from dotenv import load_dotenv
import streamlit as st
from time import sleep
from stqdm import stqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from st_chat_message import message  # for Chat-style message component

# Load from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env_region = os.getenv("PINECONE_ENVIRONMENT") 

# Initializing Pinecone client
pc = Pinecone(api_key=api_key)
index_name = "imdb-movie-vecs"
index = pc.Index(index_name)


model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("🎬 Semantic Movie Recommender Chat")
st.write("Enter a sentence describing the kind of movie you would like to watch:")

# Get user query
user_query = st.text_input("Your movie description:")

if user_query:
    # Show the user message 
    message(user_query, is_user=True)
    
    # Show a progress bar to simulate processing
    with st.spinner("Processing your request..."):
        for _ in stqdm(range(10), desc="Generating recommendations", mininterval=0.1):
            sleep(0.1)
    
    # Encode query to get vector representation
    query_vec = model.encode([user_query]).tolist()
    
    # Query Pinecone for top 10 similar movies
    result = index.query(
        namespace="default",
        vector=query_vec,
        top_k=10,
        include_metadata=True,
        include_values=False
    )
    
    # Display recommendations as chat messages
    if result and result.get("matches"):
        st.subheader("Top Recommendations:")
        for match in result["matches"]:
            meta = match["metadata"]
            rec_message = (
                f"**Title:** {meta['title']}\n\n"
                f"**Genre:** {meta['genre']}\n\n"
                f"**Overview:** {meta['overview']}\n\n"
                f"**Similarity Score:** {match['score']:.3f}"
            )
            message(rec_message, is_user=False)
    else:
        message("No recommendations found. Please try a different description.", is_user=False)

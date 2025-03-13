import weaviate
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize Weaviate client
client = weaviate.Client(
    url="http://localhost:8080"  # Assuming Weaviate is running locally on port 8080
)


def get_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Example model
        input=text
    )
    return response['data'][0]['embedding']

# OR, you can use Sentence Transformers model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the chunk from a .txt file
with open('path_to_your_chunk.txt', 'r') as file:
    text_chunk = file.read()

# Get embedding for the text chunk
embedding = get_openai_embedding(text_chunk)  # Use this for OpenAI
# embedding = model.encode(text_chunk)  # Use this for Sentence Transformers

# Create schema (if not already created)
class_obj = {
    "classes": [
        {
            "class": "TextChunk",
            "vectorizer": "text2vec-openai",  # Use the correct vectorizer for your model
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                },
            ],
        }
    ]
}

client.schema.create(class_obj)

# Insert the chunk into Weaviate
data_object = {
    "content": text_chunk,
    "vector": embedding
}

# Add the object to Weaviate
client.data_object.create(data_object, class_name="TextChunk")

print("Successfully added chunk to Weaviate.")

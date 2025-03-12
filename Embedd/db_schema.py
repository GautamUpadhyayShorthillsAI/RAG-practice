# import weaviate
# from sentence_transformers import SentenceTransformer
# import os

# # Create a Weaviate client using the v4 API (note the change here)
# client = weaviate.connect_to_custom(
#     http_host="localhost",
#     http_port=8080,
#     http_secure=False,
#     grpc_host="localhost",
#     grpc_port=50051,
#     grpc_secure=False
# )


# # Define the schema for storing text chunks
# class_obj = {
#     "class": "TextChunk",  # Name of the class
#     "vectorizer": "none",  # We will provide our own vectors
#     "properties": [
#         {
#             "name": "text",  # Field to store the text
#             "dataType": ["text"],
#         },
#         {
#             "name": "embedding",  # Field to store the vector
#             "dataType": ["number[]"],  # Array of numbers representing the vector
#         },
#     ],
# }

# # Create the schema in Weaviate
# client.schema.create_class(class_obj)

# # Function to load the text chunks from files
# def load_chunks_from_directory(directory_path):
#     chunks = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith('.txt'):
#             with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
#                 chunks.append(f.read())
#     return chunks

# # Load and embed the text chunks
# model = SentenceTransformer('all-MiniLM-L6-v2')
# chunks = load_chunks_from_directory('NCERT_downloads/extracted_texts')  # Update the path to where your .txt files are stored
# embeddings = model.encode(chunks)

# # Store the text chunks and their embeddings in Weaviate
# for i, chunk in enumerate(chunks):
#     data_object = {
#         "text": chunk,
#         "embedding": embeddings[i].tolist(),
#     }
#     client.data_object.create(data_object, class_name="TextChunk")

# # Function to search for relevant text chunks based on a query
# def search_weaviate(query, client, top_k=5):
#     query_embedding = model.encode([query])[0]
#     result = client.query.get("TextChunk", ["text", "_additional {distance}"]) \
#                         .with_near_vector({"vector": query_embedding.tolist()}) \
#                         .with_limit(top_k) \
#                         .do()
#     return result

# # Example search query
# query = "What is matter?"
# results = search_weaviate(query, client)

# # Print search results
# print("Top 5 relevant text chunks:")
# for result in results['data']['Get']['TextChunk']:
#     print(f"Text: {result['text']}")
#     print(f"Distance: {result['_additional']['distance']}")
#     print("-----")



import weaviate

# Replace with your Docker instance's host and port (default is 127.0.0.1:8080)
client = client = weaviate.connect_to_local()

# Test the connection
print(client.is_ready())

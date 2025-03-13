import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Folder containing the cleaned text files
input_folder = 'NCERT_downloads/cleaned_texts'
output_folder = 'chunks_output'

# Create a directory to store the chunked files (if it doesn't exist)
os.makedirs(output_folder, exist_ok=True)

# Initialize the RecursiveCharacterTextSplitter with chunk size 1000 and overlap 100
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # The maximum size of each chunk
    chunk_overlap=100  # The number of characters to overlap between chunks
)

# Loop through all .txt files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):  # Process only .txt files
        file_path = os.path.join(input_folder, file_name)

        # Read the content of the file
        with open(file_path, 'r') as f:
            text = f.read()

        # Split the text into chunks
        chunks = text_splitter.split_text(text)

        # Save each chunk as a separate .txt file
        for idx, chunk in enumerate(chunks):
            # Create a new file name for each chunk
            chunk_file_name = f"{os.path.splitext(file_name)[0]}_chunk_{idx + 1}.txt"
            chunk_file_path = os.path.join(output_folder, chunk_file_name)

            # Save the chunk to the new file
            with open(chunk_file_path, 'w') as chunk_file:
                chunk_file.write(chunk)

            print(f"Saved Chunk {idx + 1} from {file_name} as {chunk_file_path}")


import os
import re

class NCERTTextCleaner:
    def __init__(self, input_txt_directory, output_cleaned_directory):
        """
        Initializes the NCERTTextCleaner with the input text directory and output cleaned text directory.
        """
        self.input_txt_directory = input_txt_directory
        self.output_cleaned_directory = output_cleaned_directory

        # Ensure the output directory exists
        if not os.path.exists(self.output_cleaned_directory):
            os.makedirs(self.output_cleaned_directory)

    def _clean_text(self, text):
        """
        Clean the extracted text by removing unnecessary elements such as headers, footers, 
        page numbers, special characters, and excessive whitespace.
        """
        # Step 1: Remove headers and footers (common patterns like 'Page', 'Chapter', etc.)
        # Regular expression to remove headers or footers with page numbers and chapter info
        text = re.sub(r'Page\s*\d+', '', text)  # Remove "Page 1", "Page 2", etc.
        text = re.sub(r'\b(Page|Pg|Chapter)\s*\w+[-]?\w*\s*\d+', '', text)  # Remove "Chapter 1", "Pg. 4", etc.
        
        # Step 2: Remove any non-alphanumeric characters except spaces (remove special characters, punctuation, etc.)
        # Keep only alphabets, digits, and spaces
        # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove any special characters except for spaces

        # Step 3: Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Step 4: Remove excessive newlines, tabs, and other irrelevant whitespace
        # text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with one
        text = text.strip()  # Remove leading and trailing spaces/newlines

        return text

    def _process_txt_files(self):
        """
        Process all text files in the input directory, clean them, and save the cleaned text
        to the output directory.
        """
        cleaned_texts = []
        for filename in os.listdir(self.input_txt_directory):
            if filename.endswith('.txt'):
                txt_path = os.path.join(self.input_txt_directory, filename)
                
                # Step 1: Read the content of the text file
                with open(txt_path, 'r', encoding='utf-8') as file:
                    raw_text = file.read()

                # Step 2: Clean the extracted text
                cleaned_text = self._clean_text(raw_text)

                # Step 3: Save the cleaned text to the list
                cleaned_texts.append((filename, cleaned_text))
                
        return cleaned_texts

    def _save_cleaned_text(self, cleaned_texts):
        """
        Save cleaned text to new text files in the output directory.
        """
        for i, (filename, cleaned_text) in enumerate(cleaned_texts):
            # Create a new filename for the cleaned text
            cleaned_filename = f'cleaned_{filename}'  # Prefix the cleaned files with 'cleaned_'
            cleaned_path = os.path.join(self.output_cleaned_directory, cleaned_filename)

            # Save cleaned text to the new file
            with open(cleaned_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)
            print(f"Saved cleaned text to {cleaned_filename}")

    def process(self):
        """
        Full processing function: Clean the text files in the input directory and save the cleaned text
        to the output directory.
        """
        # Step 1: Process the .txt files (clean them)
        cleaned_texts = self._process_txt_files()

        # Step 2: Save the cleaned text to new files
        self._save_cleaned_text(cleaned_texts)
        print("Text cleaning and saving complete.")


# Driver Code
if __name__ == "__main__":
    # Specify the directories for the text files and the output cleaned text directory
    input_txt_directory = 'Extract text'  # Path where your extracted .txt files are located
    output_cleaned_directory = 'Extract text'  # Path where you want the cleaned .txt files saved

    # Initialize the NCERTTextCleaner class
    cleaner = NCERTTextCleaner(input_txt_directory, output_cleaned_directory)

    # Process the text files: Clean them and save the cleaned files
    cleaner.process()

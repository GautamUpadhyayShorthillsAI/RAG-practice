import os
import fitz  # PyMuPDF

class PDFTextExtractor:
    def __init__(self, extracted_folder, output_folder):
        self.extracted_folder = extracted_folder
        self.output_folder = output_folder

        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file."""
        try:
            # Open the PDF
            pdf_document = fitz.open(pdf_path)

            # Initialize an empty string to hold the extracted text
            text = ""

            # Iterate through each page of the PDF
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)  # Load each page
                text += page.get_text()  # Extract text from the page

            return text

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None

    def save_text_to_file(self, text, output_path):
        """Save the extracted text to a text file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            print(f"Text saved to {output_path}")
        except Exception as e:
            print(f"Error saving text to {output_path}: {e}")

    def process_pdfs(self):
        """Process all PDFs in the extracted folder."""
        # List all files in the extracted folder
        files = os.listdir(self.extracted_folder)

        for file_name in files:
            # Check if the file is a PDF
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(self.extracted_folder, file_name)
                print(f"Processing {pdf_path}...")

                # Extract text from the PDF
                text = self.extract_text_from_pdf(pdf_path)

                if text:
                    # Define output path for the extracted text
                    output_path = os.path.join(self.output_folder, file_name.replace(".pdf", ".txt"))
                    self.save_text_to_file(text, output_path)

# Driver Code
if __name__ == "__main__":
    # Define the folder containing extracted PDFs
    extracted_folder = os.path.join(os.getcwd(), "NCERT_downloads", "extracted_books")

    # Define the folder where you want to save the extracted text files
    output_folder = os.path.join(os.getcwd(), "NCERT_downloads", "extracted_texts")

    # Initialize the PDFTextExtractor class
    text_extractor = PDFTextExtractor(extracted_folder, output_folder)

    # Process all PDFs in the extracted folder
    text_extractor.process_pdfs()

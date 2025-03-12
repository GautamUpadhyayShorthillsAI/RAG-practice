import os
import time
import zipfile
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.action_chains import ActionChains

class NCERTBookDownloader:
    def __init__(self, download_folder=None):
        # Set default download folder path if not provided
        self.download_folder = download_folder or os.path.join(os.getcwd(), "NCERT_downloads")
        self.driver = None

        # Ensure the download folder exists
        if not os.path.exists(self.download_folder):
            os.makedirs(self.download_folder)

    def _initialize_driver(self):
        """Initialize the WebDriver with the required options."""
        options = webdriver.ChromeOptions()

        # Set the download folder in Chrome preferences
        preferences = {
            "download.default_directory": self.download_folder,  # Custom download folder
            "download.prompt_for_download": False,  # Disable download prompt
            "directory_upgrade": True
        }

        options.add_experimental_option("prefs", preferences)

        # Optional: Run in headless mode (without a GUI)
        options.add_argument("--headless")

        # Initialize the WebDriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def _wait_for_download(self):
        """Wait until the download is completed (i.e., .crdownload disappears)."""
        download_in_progress = True
        while download_in_progress:
            # List the files in the download folder
            downloaded_files = os.listdir(self.download_folder)
            zip_file = None

            # Check for the zip file with .crdownload extension
            for file in downloaded_files:
                if file.endswith(".zip.crdownload"):
                    print("Download in progress, waiting for completion...")
                    time.sleep(2)  # Wait before checking again
                    break
                elif file.endswith(".zip"):  # Check for the actual zip file
                    zip_file = os.path.join(self.download_folder, file)
                    download_in_progress = False
                    break
            if not download_in_progress:
                break

        return zip_file

    def _unzip_file(self, zip_file):
        """Unzip the downloaded zip file into a specific folder."""
        extracted_folder = os.path.join(self.download_folder, "extracted_books")
        if not os.path.exists(extracted_folder):
            os.makedirs(extracted_folder)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

        print(f"Unzipped the file to {extracted_folder}")

        # Optionally, delete the zip file after extraction
        os.remove(zip_file)
        print(f"Deleted the original zip file: {zip_file}")

    def download_and_unzip(self, url):
        """Download and unzip the book for the given URL."""
        # Open the NCERT page
        self.driver.get(url)

        # Allow the page to load
        time.sleep(3)

        try:
            # Locate the "Download Complete Book" <a> tag by text
            download_link = self.driver.find_element(By.LINK_TEXT, "Download complete book")
            
            # Scroll to the download link (optional)
            actions = ActionChains(self.driver)
            actions.move_to_element(download_link).perform()
            time.sleep(1)  # Give it time to scroll into view

            # Click the download link
            download_link.click()
            print(f"Clicking 'Download Complete Book' on {url}...")

            # Wait for the download to complete and get the zip file path
            zip_file = self._wait_for_download()

            if zip_file:
                # Unzip the file
                self._unzip_file(zip_file)
            else:
                print("No zip file found in the download folder.")
        
        except Exception as e:
            print(f"Error: {str(e)}")

    def process_urls(self, urls):
        """Process multiple URLs to download and unzip books."""
        for url in urls:
            self.download_and_unzip(url)

    def close_driver(self):
        """Close the WebDriver instance."""
        if self.driver:
            self.driver.quit()

# Driver Code
if __name__ == "__main__":
    # Define the URLs to process
    urls = [
        'https://ncert.nic.in/textbook.php?fecu1=0-12',
        'https://ncert.nic.in/textbook.php?gesc1=0-13',
        'https://ncert.nic.in/textbook.php?hesc1=0-13',
        'https://ncert.nic.in/textbook.php?iesc1=0-12',
        'https://ncert.nic.in/textbook.php?jesc1=0-13'
    ]

    # Initialize the NCERTBookDownloader class
    downloader = NCERTBookDownloader()

    # Initialize the WebDriver
    downloader._initialize_driver()

    # Process all URLs to download and unzip books
    downloader.process_urls(urls)

    # Close the WebDriver after processing
    downloader.close_driver()

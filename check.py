import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# URL of the website containing PDF links
website_url = "https://kiit.ac.in/academics/syllabus/"  # Change this to the target website

# Folder to save PDFs
download_folder = "pdf_downloads"
os.makedirs(download_folder, exist_ok=True)

# Get website content
response = requests.get(website_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all PDF links
    pdf_links = [urljoin(website_url, a["href"]) for a in soup.find_all("a", href=True) if a["href"].endswith(".pdf")]

    if not pdf_links:
        print("No PDF links found!")
    else:
        print(f"Found {len(pdf_links)} PDFs. Downloading...")

        for pdf_url in pdf_links:
            pdf_name = os.path.join(download_folder, pdf_url.split("/")[-1])

            # Download PDF
            pdf_response = requests.get(pdf_url)
            with open(pdf_name, "wb") as pdf_file:
                pdf_file.write(pdf_response.content)

            print(f"Downloaded: {pdf_name}")
else:
    print("Failed to access the website!")

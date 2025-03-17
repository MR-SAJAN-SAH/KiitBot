import pymupdf # PyMuPDF for text extraction
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# ✅ Set Tesseract path (Windows users: change this if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\KIIT\TESSER\tesseract.exe"


def extract_text_from_pdf(pdf_path, output_img_folder):
    """Extracts text from a given PDF, using OCR for image-based PDFs."""
    doc = pymupdf.open(pdf_path)
    text = ""

    # ✅ Extract text from each page
    for page in doc:
        page_text = page.get_text("text")  # Get raw text
        text += page_text + "\n"

    # ✅ If no text is found, process as an image-based PDF
    if not text.strip():
        print(f"🔍 {pdf_path} seems image-based. Using OCR...")
        os.makedirs(output_img_folder, exist_ok=True)  # Create folder for extracted images

        images = convert_from_path(pdf_path)  # Convert PDF pages to images
        for i, img in enumerate(images):
            img_filename = os.path.join(output_img_folder, f"{os.path.basename(pdf_path)}_page_{i + 1}.png")
            img.save(img_filename, "PNG")  # Save extracted images (optional)

            # ✅ Perform OCR on the extracted image
            text += pytesseract.image_to_string(img) + "\n"

    return text


# ✅ Process multiple PDFs
pdf_folder = "gmail_pdfs/"
output_text_folder = "text_files/"  # Folder for extracted text
output_img_folder = "image_files/"  # Folder for extracted images from image-based PDFs
os.makedirs(output_text_folder, exist_ok=True)

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        extracted_text = extract_text_from_pdf(pdf_path, output_img_folder)

        # ✅ Save extracted text
        text_filename = os.path.join(output_text_folder, pdf_file.replace(".pdf", ".txt"))
        with open(text_filename, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

        print(f"✅ Extracted text from {pdf_file} and saved to {text_filename}")

print("✅ All PDFs processed successfully!")

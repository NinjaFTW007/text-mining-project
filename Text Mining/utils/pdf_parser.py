import pypdf

def extract_text_from_file(uploaded_file) -> str:
    """Extracts strings from an uploaded file (TXT or PDF)."""
    filename = uploaded_file.name.lower()
    
    if filename.endswith(".pdf"):
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        # Reset the file pointer just in case it's re-read elsewhere
        uploaded_file.seek(0)
        return text.strip()
    else:
        # Assume standard text file
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        uploaded_file.seek(0)
        return content

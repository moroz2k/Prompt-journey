from pypdf import PdfReader
reader = PdfReader("D:\\prompt\\test.pdf")
for page in reader.pages: print(page.extract_text())

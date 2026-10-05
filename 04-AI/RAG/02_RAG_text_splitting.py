from dotenv import load_dotenv
load_dotenv()


from langchain_text_splitters import RecursiveCharacterTextSplitter
texts = "Good morning everyone,Today, I want to talk about something that is probably on the mind of every student, developer, and IT professional:"
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 50, chunk_overlap = 20)
text_chunks = text_splitter.split_text(texts)  # For text
#  we also have split_documents() for document split
print(text_chunks)
# ['Good morning everyone,Today, I want to talk about', 
# 'want to talk about something that is probably on', 
# 'that is probably on the mind ofevery student,', 
# 'of every student, developer, and IT professional:']

# For Pdf Splitter
# PDF Document Loader
from langchain_community.document_loaders import PyPDFLoader
pdf_file_loader = PyPDFLoader(
    "rag_pdf_loader_practice.pdf"
) # Give Path of the text File as arg in string 
pdf_data_docs = pdf_file_loader.load()  # Load Data 
print(len(pdf_data_docs))
pdf_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
pdf_data_docs = pdf_file_loader.load()  # Load Data 
pdf_chunks = pdf_splitter.split_documents(pdf_data_docs)
print(len(pdf_chunks))
print(pdf_chunks[0].page_content)





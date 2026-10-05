from dotenv import load_dotenv
load_dotenv()

# Text Document Loader
from langchain_community.document_loaders import TextLoader
text_file_loader = TextLoader(
    "speech.txt",
    encoding="utf-8"
) # Give Path of the text File as arg in string 
text_data_docs = text_file_loader.load()  # Load Data 
# print(len(text_data_docs))
oneDoc_text = text_data_docs[0]
# print(oneDoc_text.page_content)


# PDF Document Loader
from langchain_community.document_loaders import PyPDFLoader
pdf_file_loader = PyPDFLoader(
    "rag_pdf_loader_practice.pdf"
) # Give Path of the text File as arg in string 
pdf_data_docs = pdf_file_loader.load()  # Load Data 
# print(len(pdf_data_docs)) # No of pages
oneDoc_pdf = pdf_data_docs[0]
# print(oneDoc_pdf.page_content)

# CSV Document Loader
from langchain_community.document_loaders import CSVLoader
csv_file_loader = CSVLoader(
    "ford.csv"
) # Give Path of the text File as arg in string 
csv_data_docs = csv_file_loader.load()  # Load Data 
# print(len(pdf_data_docs)) # No of pages
oneDoc_csv = csv_data_docs[:5] # First 5 
# print(oneDoc_csv)

# Web page Loader
from langchain_community.document_loaders import WebBaseLoader
web_page_loader = WebBaseLoader(web_path="https://www.cdac.in/index.aspx?id=about")
web_data_docs = web_page_loader.load()
oneDoc_web = web_data_docs[0].page_content
# print(oneDoc_web)


#  We have many document loaders in Langchain like for youtube, whatsapp chat , wikipedia etc 

# WikiPedia Loader
from langchain_community.document_loaders import WikipediaLoader
wiki_loader = WikipediaLoader(query="Gen AI", load_max_docs=1)
wiki_data_docs = wiki_loader.load()
oneDoc_wiki = wiki_data_docs[0].page_content
# print(oneDoc_wiki)




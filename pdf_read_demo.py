from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("local_doc/vsap.pdf")
pages = loader.load_and_split()

for i, page in enumerate(pages):
    print(f"========= 第 {i} 段输出 ========")
    print(page.page_content)

#this is to split the text of the files into chunks and by utilizing certain separators

from langchain.text_splitter import RecursiveCharacterTextSplitter



def split_documents(documents, chunk_size=800, chunk_overlap=100):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs

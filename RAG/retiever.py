from tools import get_file_text
from pathlib import Path

def get_all_documents():
    documents = {}
    folder = Path("data")
    for file in folder.glob("*.txt"):
        documents[file.name] = file.read_text(encoding="utf-8")
    return documents

def search_text(question: str):
    documents = get_all_documents()
    question = question.lower()
    for file_name , content in documents.items():
        if question in content.lower():
            return file_name, content
    return None, None


    
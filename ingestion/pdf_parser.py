from llama_index.core import SimpleDirectoryReader


def parse_pdf(file_path: str | Path) -> list[Document]
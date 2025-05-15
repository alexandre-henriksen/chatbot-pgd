import os
from dotenv import load_dotenv

def load_config():
    """Carrega variáveis de ambiente e configurações."""
    load_dotenv()

    config = {
        "hf_token": "HF_TOKEN",
        "db_path": "./chroma_db",
        "docs_dir": "docs",
        "chunk_size": 1500,
        "chunk_overlap": 200,
        "similarity_threshold": 0.65,
        "max_tokens": 4096,
        "temperature": 0,
        "keywords": ["api", "módulos", "pdg", "painel", "plano", "trabalho", "entrega", "registro", "execução", "gestão", "desempenho"],
        "csv_options": {
            "default_delimiter": ",",
            "default_quotechar": '"',
            "encoding": "utf-8",
            "handle_headers": True,
        }
    }

    return config

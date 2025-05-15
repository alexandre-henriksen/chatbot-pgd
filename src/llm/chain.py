from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEndpoint

def get_memory(k=3):
    """Configura a memória de conversação."""
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        input_key="question"
    )

def get_llm(config):
    return HuggingFaceEndpoint(
        endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=config["hf_token"],
        max_new_tokens=config["max_tokens"],
        temperature=config["temperature"],
    )

def setup_chain(llm, prompt_template, memory):
    """Configura a cadeia de processamento LLM."""
    return LLMChain(
        llm=llm,
        prompt=prompt_template,
        memory=memory,
        verbose=False
    )

def ask_question(question, retriever, llm_chain, embeddings, keywords, threshold=0.65, return_sources=False):
    """Processa uma pergunta e retorna a resposta e opcionalmente os documentos fonte."""
    from src.retrieval.embeddings import filter_relevant_documents

    docs = retriever.get_relevant_documents(question)
    print(f"[DEBUG] Documentos brutos recuperados: {len(docs)}")

    relevant_docs = filter_relevant_documents(question, docs, embeddings, keywords, threshold)
    context = "\n".join(doc.page_content for doc in relevant_docs)
    print(f"[DEBUG] Documentos filtrados (primeiros 1000 caracteres):\n{context[:1000]}\n")

    result = llm_chain({
        "question": question,
        "context": context,
    })

    # Retorna a resposta e opcionalmente os documentos relevantes
    if return_sources:
        return result["text"], relevant_docs
    else:
        return result["text"]

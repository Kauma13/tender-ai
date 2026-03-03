'''All LLM/embedding client constructors'''

import os
from app.core.config import settings
from dotenv import load_dotenv
load_dotenv()

def get_embeddings():
    """
    Returns a LangChain embeddings object used by Chroma.
    """
    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embeddings_model)

    raise ValueError(f"Unsupported embeddings_provider={settings.embeddings_provider}")

# def get_classifier_llm():
#     """
#     LLM used for domain + risk classification.
#     """
#     if settings.domain_classifier_provider == "openai":
#         from langchain_openai import ChatOpenAI
#         return ChatOpenAI(model=settings.domain_classifier_model, temperature=0)

#     raise ValueError(f"Unsupported classifier provider={settings.domain_classifier_provider}")


def get_generator_llm():
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, temperature=0.2)

    raise ValueError(f"Unsupported LLM provider={settings.llm_provider}")
    

def get_llm():
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.llm_model, temperature=0)

    raise ValueError(f"Unsupported LLM provider={settings.llm_provider}")

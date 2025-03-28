from .mistral7b import query_mistral7b
from .llama8b import query_llama8b
from .llama3b import query_llama3b
from .llama1b import query_llama1b

QUERY = dict(
  Mistral7B=query_mistral7b,
  Llama8B=query_llama8b,
  Llama3B=query_llama3b,
  Llama1B=query_llama1b
)
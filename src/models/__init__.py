from .mistral7b import query_mistral7b
from .llama8b import query_llama8b
from .llama3b import query_llama3b
from .llama1b import query_llama1b

QUERY = dict(
  mistral7b=query_mistral7b,
  llama8b=query_llama8b,
  llama3b=query_llama3b,
  llama1b=query_llama1b,
)
from .mistral7b import query_mistral7b
from .llama8b import query_llama8b

QUERY = dict(
  Mistral7B=query_mistral7b,
  Llama8b=query_llama8b
)
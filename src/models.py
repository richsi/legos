from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Mistral7BWrapper:
  def __init__(self):
    self.model_name = "mistralai/Mistral-7B-Instruct-v0.1" 
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.llm = AutoModelForCausalLM.from_pretrained(
      self.model_name,
      torch_dtype=torch.float16,
      device_map="auto"
    )
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.llm.to(self.device)
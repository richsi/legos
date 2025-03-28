_llama_cache = {}

def get_llama3b_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global _llama_cache 
    if "llama_model" not in _llama_cache:
        model_id = "meta-llama/Llama-3.2-3b-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _llama_cache["llama_tokenizer"] = tokenizer
        _llama_cache["llama_model"] = model
    return _llama_cache["llama_tokenizer"], _llama_cache["llama_model"]


def query_llama3b(full_prompt: str) -> str:
    import torch
    tokenizer, model = get_llama3b_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoded = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        # truncation=True
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=input_ids.shape[1] + 200,
        do_sample=False
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text
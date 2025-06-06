_mistral_cache = {}

def get_mistral_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    global _mistral_cache  
    if "mistral_model" not in _mistral_cache :
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        _mistral_cache ["mistral_tokenizer"] = tokenizer
        _mistral_cache ["mistral_model"] = model
    return _mistral_cache ["mistral_tokenizer"], _mistral_cache ["mistral_model"]


def query_mistral7b(full_prompt: str, sc: bool = False):
    import torch
    tokenizer, model = get_mistral_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoded = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        # truncation=True
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    # input_ids = encoded.input_ids
    # attention_mask = encoded.attention_mask
    
    if sc:
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            # max_new_tokens=1000,
            max_length=input_ids.shape[1] + 5000,
            do_sample=True,
            num_return_sequences=3,
            temperature=0.5, top_k=10, top_p=1.0
        )
        # print("output_ids", len(output_ids))
        output_texts = [tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]
        # print("output_texts", len(output_texts))
        return output_texts
    else:
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            # max_new_tokens=1000,
            max_length=input_ids.shape[1] + 5000,
            do_sample=True
        )
        # print(len(output_ids))
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
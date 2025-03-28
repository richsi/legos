def query_mistral7b(full_prompt: str) -> str:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    
    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 200,  # adjust as needed
        do_sample=False
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)

print(tokenizer.bos_token)
print(tokenizer.eos_token)

print(tokenizer.decode([151643, 151644]))

print(tokenizer.bos_id)
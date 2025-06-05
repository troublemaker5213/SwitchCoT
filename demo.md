## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "switchcot-7B"

llm, tokenizer = load_hf_lm_and_tokenizer(
    model_name_or_path=args.model_name_or_path,
    load_in_half=True,
    use_fast_tokenizer=True,
    use_safetensors=args.use_safetensors,
)

instruction = "Please reason step by step, and put your final answer within \\boxed{}."
prompt = "The arithmetic mean of 7, 2, $x$ and 10 is 9. What is the value of $x$?"
#prompt = "What is the smallest positive perfect cube that can be written as the sum of three consecutive integers?"
# prompt = "How many r's are in the word \"strawberry\""

messages = [
    {"role": "user", "content": f"{instruction}\n{prompt}"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = generate_completions(
    model=llm,
    tokenizer=tokenizer,
    prompts=model_inputs,
    max_new_tokens=args.max_tokens_per_call,
    batch_size=4,
    stop_id_sequences=stop_words,
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
num_tokens = len(generated_ids[0])

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

think_mode = ("<think>" in response)

print(text+response)
print(f"\nThink Mode: {think_mode}")
print(f"Number of tokens: {num_tokens}")
```


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
 
# 1. Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # GPT-2 model (can be replaced with a larger model like gpt2-medium)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
 
model.eval()  # Set model to evaluation mode
 
# 2. Generate controlled text based on a given tone or style
def generate_controlled_text(prompt, tone="formal", max_length=150, temperature=0.7, top_p=0.9):
    # Condition the input based on tone (e.g., formal or informal)
    conditioned_prompt = f"Write a {tone} paragraph: {prompt}"
    inputs = tokenizer.encode(conditioned_prompt, return_tensors='pt')
    
    # Generate text with the model
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1,
                                 temperature=temperature, top_p=top_p, no_repeat_ngram_size=2)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
 
# 3. Example controlled text generation
prompt = "The company's recent product launch"
generated_text_formal = generate_controlled_text(prompt, tone="formal")
generated_text_informal = generate_controlled_text(prompt, tone="informal")
 
print("Generated Formal Text:")
print(generated_text_formal)
 
print("\nGenerated Informal Text:")
print(generated_text_informal)

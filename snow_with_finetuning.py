import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_base_model():
    """Load the base TinyLlama model."""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading base {model_name} model...")
    device = "cpu"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model optimized for CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        low_cpu_mem_usage=True
    )
    model.to(device)
    
    return model, tokenizer

def load_finetuned_model():
    """Load the fine-tuned model."""
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    finetuned_path = "D:\\finetune-with-windsurf\\snow_finetuned_model"
    
    # Check if fine-tuned model exists
    if not os.path.exists(finetuned_path):
        print(f"Fine-tuned model not found at {finetuned_path}")
        return None, None
    
    try:
        print(f"Loading fine-tuned model from {finetuned_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(finetuned_path)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load fine-tuned model adapters
        model = PeftModel.from_pretrained(base_model, finetuned_path)
        model.to("cpu")
        
        print("Fine-tuned model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None, None

def generate_response(model, tokenizer, question, debug=False):
    """Generate a response from the model for the given question."""
    if model is None or tokenizer is None:
        return "Model not available"
    
    # Format prompt for TinyLlama
    system_instruction = "You are a helpful assistant. Please provide accurate and concise information."
    prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    if debug:
        print(f"\nInput prompt:\n{prompt}")
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate a response with proper attention mask handling
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Explicitly pass attention_mask to prevent warnings
            max_new_tokens=150,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if debug:
        print(f"\nRaw generated text:\n{generated_text}\n")
    
    # Extract just the assistant's response
    try:
        parts = generated_text.split("<|im_start|>assistant\n")
        if len(parts) > 1:
            assistant_response = parts[1].split("<|im_end|>")[0]
        else:
            # Fallback if the expected format is not found
            assistant_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except IndexError:
        # Fallback if the expected format is not found
        assistant_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    return assistant_response.strip()

def main():
    print("Comparing TinyLlama base model vs. snow fine-tuned model responses")
    print("=" * 70)
    
    # Load models
    base_model, base_tokenizer = load_base_model()
    finetuned_model, finetuned_tokenizer = load_finetuned_model()
    
    # Question to ask both models about snow in summer
    snow_question = "Does it snow in summer?"
    print(f"\nQuestion: {snow_question}\n")
    
    # Generate responses with debug information
    print("Generating response from base model...")
    base_response = generate_response(base_model, base_tokenizer, snow_question, debug=True)
    
    print("\nGenerating response from fine-tuned model...")
    finetuned_response = generate_response(finetuned_model, finetuned_tokenizer, snow_question, debug=True)
    
    # Try with a variation of the snow question
    variant_question = "Can you see snow during summer months?"
    print(f"\n\nTrying a variation question: '{variant_question}'")
    print("Generating response from fine-tuned model...")
    variant_response = generate_response(finetuned_model, finetuned_tokenizer, variant_question, debug=True)
    
    # Display results separately
    print("\n\nBase TinyLlama Model Response (Snow Question):")
    print("=" * 70)
    print(base_response)
    
    print("\n\nFine-tuned TinyLlama Model Response (Snow Question):")
    print("=" * 70)
    print(finetuned_response)
    
    print("\n\nFine-tuned TinyLlama Model Response (Variation Question):")
    print("=" * 70)
    print(variant_response)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")




from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,     
                                             device_map="auto",  # Automatically assign parts of the model to CPU and GPU
                                            offload_folder="./offload",  # Directory to offload to CPU
                                            )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(torch.__version__)

# Create a pipeline
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id,  # Avoid padding warnings
                truncation=True,  # Explicitly enable truncation
                temperature=0.8)

# Initialize conversation history (internal, not displayed)
conversation_history = []
print()

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Add the user input to the conversation history
    conversation_history.append(f"Human: {user_input}")

    # Build the prompt dynamically 
    context = "\n".join(conversation_history[-6:])  # Limit history size
    prompt = f"The following is a snarky conversation between a human and an AI:\n{context}\nAI:"

    # Generate the response
    response = pipe(prompt)[0]["generated_text"]

    # Extract only the new part of the AI's response
    ai_response = response[len(prompt):].strip().split("\n")[0]  # Get first AI response
    print(f"AI: {ai_response}")
    print()

    # Add the AI's response to the history
    conversation_history.append(f"AI: {ai_response}")
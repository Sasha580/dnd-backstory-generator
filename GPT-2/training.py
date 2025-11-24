import torch
import re
from custom_dataset import CustomDataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer


MAX_LENGTH = 200
BATCH_SIZE = 4
EPOCHS = 3
SAVE_INTERVAL = 1


def normalize_text(text):
    text = re.sub(r"[ÓÒ]", '"', text)
    text = re.sub(r"Õ", "'", text)
    text = re.sub(r"\r\r", "", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\. ", ".\n", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"_", "V", text)

    # text_with_newlines = '\n'.join([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])

    return text


def train(custom_dataset, model, optimizer):
    """
    Train the model with checkpoint saving during the epoch.

    Args:
        custom_dataset (DataLoader): DataLoader providing batches of input_ids and attention_mask.
        model (nn.Module): The model to train.
        optimizer (Optimizer): The optimizer used for updating model weights.
    """

    for i in range(EPOCHS):
        print(f"Starting epoch {i + 1}/{EPOCHS}")
        for batch_idx, (input_ids, attention_mask) in enumerate(custom_dataset):
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:  # Show progress every 10 batches
                print(f"Epoch {i + 1}, Batch {batch_idx + 1}/{len(custom_dataset)}: Loss = {loss.item():.4f}")

        # Save the model at the end of the epoch
        torch.save(model.state_dict(), f"epoch_{i}.pt")
        print(f"Model saved at the end of epoch {i}")


def predict(name, race, char_class, model, tokenizer):
    # Create the input prompt with the character's details
    prompt = (
        "Generate Backstory based on the following information:\n"
        f"Character Name: {name}\n"
        f"Character Race: {race}\n"
        f"Character Class: {char_class}"
    )

    # Tokenize the prompt for the model
    my_input = tokenizer(prompt, return_tensors="pt")

    # Generate the model output
    output = model.generate(**my_input, max_length=200)

    # Decode the output into a readable string
    decoded_output = tokenizer.decode(output[0])

    return decoded_output


def main():
    dataset = load_dataset("MohamedRashad/dnd_characters_backstories")

    filtered_data = [
        "<startofstring> "
        + normalize_text(x["text"])
        + "<bot>: "
        + normalize_text(x["target"])
        + " <endofstring>"
        for x in dataset['train']
        if x["text"] is not None
        and x["target"] is not None
    ]

    # Remove an unwanted element from filtered data
    filtered_data.pop(1)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Add special tokens to tokenizer
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_tokens(["<bot>: ", "<startofstring>", "<endofstring>"])
    model.resize_token_embeddings(len(tokenizer))

    # Create PyTorch Dataset and DataLoader
    custom_dataset = CustomDataset(filtered_data, tokenizer)
    dataloader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optim = Adam(model.parameters())

    # Uncomment to train the model
    # train(dataloader, model, optim)


    # Load custom weights into the model
    custom_weights_path = "epoch.pt"
    state_dict = torch.load(custom_weights_path)
    model.load_state_dict(state_dict)

    print('\n\n Predictions')
    print(predict('Kropus', 'Tiefling', 'Mage', model, tokenizer))


if __name__ == "__main__":
    main()






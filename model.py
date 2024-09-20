import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

from TextDataset import TextDataset


class GPT2FineTuner:
    def __init__(self, model_dir='dbmdz/german-gpt2'):
        # Load the tokenizer and model using the full model name
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelWithLMHead.from_pretrained(model_dir)

    def train(self, inputs, targets, epochs=3, batch_size=8, lr=5e-5):
        # Create a dataset from the padded inputs and targets
        train_dataset = TextDataset(inputs, targets)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()

                # Ensure the batch is in the correct format
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     labels=batch['labels']
                                     )

                loss = outputs.loss
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from biencoder import (
    BiEncoder,
    train_dataloader,
    train_labels,
    validation_dataloader,
    validation_dataset,
    vocab_size,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
model = BiEncoder(
    embedding_dim=vocab_size, hidden_size=512, output_size=len(set(train_labels))
).to(device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Wrap the train_dataloader with tqdm for a progress bar
for epoch in range(150):
    model.train()
    # Use tqdm to add a progress bar to the training loop
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{150}"):
        text, labels = batch
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(text)

        # Reshape outputs to match the expected shape
        outputs = outputs.reshape(-1, len(set(train_labels)))
        labels = labels.view(-1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in tqdm(validation_dataloader, desc="Validation"):
        text, labels = batch
        outputs = model(text)
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (
            torch.sum(predictions == labels).item()
        )

accuracy = correct_predictions / len(validation_dataset)
print(f"Accuracy: {accuracy}")

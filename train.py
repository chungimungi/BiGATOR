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

device = torch.device("cuda")

model = BiEncoder(
    embedding_dim=vocab_size,
    hidden_size=256,
    output_size=len(set(train_labels))
).to(device)
#model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{10}"):
        text, labels = batch
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(text)

        print(f"Outputs shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")

        # Ensure the number of classes is correct
        num_classes = len(set(train_labels))
        if outputs.size(1) != num_classes:
            outputs = outputs[:, :num_classes]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo

from biencoder import BiEncoder, train_labels, vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
model = BiEncoder(
    embedding_dim=vocab_size,
    hidden_size=512,
    output_size=len(set(train_labels)),
).to(device)
model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

torchinfo.summary(model)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from lambada_load import test_dataset, train_dataset, validation_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiEncoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_attention_heads=4
    ):
        super().__init__()

        # Encoder for text
        self.text_encoder = nn.Sequential(
            nn.Embedding(input_size, hidden_size),
            nn.LSTM(hidden_size, hidden_size, batch_first=True),
        )
        
        self.embedding_dim = hidden_size

        # Self-Attention Layer
        self.self_attention = nn.MultiheadAttention(
            self.embedding_dim, num_attention_heads
        )

        # Linear layer for classification
        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, text):
        _, (text_hidden, _) = self.text_encoder(text)

        # Use self-attention on the text_hidden
        text_hidden = text_hidden.permute(
            1, 0, 2
        )  # Change the order of dimensions
        text_hidden, _ = self.self_attention(
            text_hidden, text_hidden, text_hidden
        )
        text_hidden = text_hidden.permute(
            1, 0, 2
        )  # Change the order of dimensions back

        # Take the mean across the sequence dimension
        text_hidden = text_hidden.mean(dim=1)

        output = self.fc(text_hidden)
        return output


train_texts = [example["text"] for example in train_dataset]
train_labels = [example["domain"] for example in train_dataset]

validation_texts = [example["text"] for example in validation_dataset]
validation_labels = [example["domain"] for example in validation_dataset]


# Tokenize and encode the training and validation data
max_len = 2500
vocab_size = 10000

# Tokenize and pad the training data
train_encodings = [
    torch.tensor([hash(word) % vocab_size for word in text.split()[:max_len]])
    for text in train_texts
]
train_encodings_padded = pad_sequence(
    train_encodings, batch_first=True, padding_value=0
).to(device)

# Tokenize and pad the validation data
validation_encodings = [
    torch.tensor([hash(word) % vocab_size for word in text.split()[:max_len]])
    for text in validation_texts
]
validation_encodings_padded = pad_sequence(
    validation_encodings, batch_first=True, padding_value=0
).to(device)

# Convert labels to tensors
train_labels = torch.tensor(
    [train_labels.index(label) for label in train_labels]
).to(device)
validation_labels = torch.tensor(
    [validation_labels.index(label) for label in validation_labels]
).to(device)

# Create DataLoader for training and validation
train_dataset = TensorDataset(train_encodings_padded, train_labels)
validation_dataset = TensorDataset(
    validation_encodings_padded, validation_labels
)
# Use DataLoader with dataset as the first argument
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2)

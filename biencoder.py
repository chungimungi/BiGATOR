import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from lambada_load import test_dataset, train_dataset, validation_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_size,
        output_size,
        num_attention_heads=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Embedding layer
        self.embedding1 = nn.Embedding(embedding_dim, hidden_size)

        # Second embedding layer
        self.embedding2 = nn.Embedding(embedding_dim, hidden_size)

        # Embedding layer
        self.embedding3 = nn.Embedding(embedding_dim, hidden_size)

        # Second embedding layer
        self.embedding4 = nn.Embedding(embedding_dim, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.lstm1 = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )

        self.lstm2 = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        

        self.embedding_dim = hidden_size

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, 256), nn.Sigmoid()
        )

        # Self-Attention Layer 1
        self.self_attention1 = nn.MultiheadAttention(
            self.embedding_dim, num_attention_heads
        )

        # Self-Attention Layer 2
        self.self_attention2 = nn.MultiheadAttention(
            self.embedding_dim, num_attention_heads
        )

        # Linear layer for classification
        self.fc = nn.Linear(self.embedding_dim, output_size)

    def forward(self, text):
        # Apply the first embedding layer
        embedded_text1 = self.embedding1(text)

        # Apply the second embedding layer
        embedded_text2 = self.embedding2(text)

        embedded_text3 = self.embedding1(text)

        embedded_text4 = self.embedding2(text)

        # Sum the outputs of both embedding layers
        embedded_text = embedded_text1 + embedded_text2 + embedded_text3 + embedded_text4

        # Apply the LSTM layer
        _, (text_hidden, _) = self.lstm(embedded_text *2)
        _, (text_hidden, _) = self.lstm1(embedded_text * 2)
        _, (text_hidden, _) = self.lstm2(embedded_text * 2)

        # Apply dropout
        text_hidden = self.dropout(text_hidden)

        # Use gating mechanism
        gate = self.gate(text_hidden.mean(dim=1).unsqueeze(1))
        text_hidden = text_hidden * gate

        # Use self-attention (Layer 1) on the text_hidden
        text_hidden = text_hidden.permute(
            1, 0, 2
        )  # Change the order of dimensions
        text_hidden, _ = self.self_attention1(
            text_hidden, text_hidden, text_hidden
        )
        text_hidden = text_hidden.permute(
            1, 0, 2
        )  # Change the order of dimensions back

        # Use self-attention (Layer 2) on the text_hidden
        text_hidden, _ = self.self_attention2(
            text_hidden, text_hidden, text_hidden
        )

        # Take the mean across the sequence dimension
        text_hidden = text_hidden.mean(dim=1)

        output = self.fc(text_hidden)
        return output


train_texts = [example["text"] for example in train_dataset]
train_labels = [example["domain"] for example in train_dataset]

validation_texts = [example["text"] for example in validation_dataset]
validation_labels = [example["domain"] for example in validation_dataset]


# Tokenize and encode the training and validation data
max_len = 5000
vocab_size = 15000

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
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=64)

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

        self.embedding1 = nn.Embedding(embedding_dim, hidden_size)
        self.embedding2 = nn.Embedding(embedding_dim, hidden_size)
        self.hidden_size = hidden_size

        # Adjust LSTM layers to account for bidirectional output
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.lstm1 = nn.LSTM(
            hidden_size * 2, hidden_size, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_size * 2, hidden_size, batch_first=True, bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2), nn.Sigmoid()
        )

        self.self_attention1 = nn.MultiheadAttention(
            hidden_size * 2, num_attention_heads
        )

        self.self_attention2 = nn.MultiheadAttention(
            hidden_size * 2, num_attention_heads
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, text):
        print(f"Input text shape: {text.shape}")

        embedded_text1 = self.embedding1(text)
        embedded_text2 = self.embedding2(text)
        embedded_text = embedded_text1 + embedded_text2

        lstm_out, _ = self.lstm(embedded_text)
        lstm_out, _ = self.lstm1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)

        lstm_out = self.dropout(lstm_out)

        # Apply gate
        gate = self.gate(lstm_out)
        lstm_out = lstm_out * gate

        # Apply attention
        lstm_out = lstm_out.permute(1, 0, 2)
        attn_out, _ = self.self_attention1(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)

        attn_out, _ = self.self_attention2(attn_out, attn_out, attn_out)

        # Global average pooling
        pooled = attn_out.mean(dim=1)

        output = self.fc(pooled)
        return output

train_texts = [example["text"] for example in train_dataset]
train_labels = [example["domain"] for example in train_dataset]

validation_texts = [example["text"] for example in validation_dataset]
validation_labels = [example["domain"] for example in validation_dataset]

max_len = 5000
vocab_size = 15000

train_encodings = [
    torch.tensor([hash(word) % vocab_size for word in text.split()[:max_len]])
    for text in train_texts
]
train_encodings_padded = pad_sequence(
    train_encodings, batch_first=True, padding_value=0
).to(device)

validation_encodings = [
    torch.tensor([hash(word) % vocab_size for word in text.split()[:max_len]])
    for text in validation_texts
]
validation_encodings_padded = pad_sequence(
    validation_encodings, batch_first=True, padding_value=0
).to(device)

train_labels = torch.tensor(
    [train_labels.index(label) for label in train_labels]
).to(device)
validation_labels = torch.tensor(
    [validation_labels.index(label) for label in validation_labels]
).to(device)

train_dataset = TensorDataset(train_encodings_padded, train_labels)
validation_dataset = TensorDataset(
    validation_encodings_padded, validation_labels
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=64)


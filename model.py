# ============================================================
# PART 3: NMT MODEL
# English -> Hindi
# Encoder-Decoder LSTM with Attention
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = None
OUTPUT_DIM = None

EMBEDDING_DIM = 256
HIDDEN_DIM = 512

NUM_LAYERS = 1

DROPOUT = 0.2


# ============================================================
# 1. ENCODER
# ============================================================

class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.2
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            input_dim,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0
            )
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, source):

        # ----------------------------------------------------
        # source shape:
        # [batch_size, source_length]
        # ----------------------------------------------------

        embedded = self.dropout(
            self.embedding(source)
        )

        # ----------------------------------------------------
        # outputs:
        # [batch_size, source_length, hidden_dim]
        #
        # hidden:
        # [num_layers, batch_size, hidden_dim]
        #
        # cell:
        # [num_layers, batch_size, hidden_dim]
        # ----------------------------------------------------

        outputs, (hidden, cell) = self.rnn(
            embedded
        )

        return (
            outputs,
            hidden,
            cell
        )


# ============================================================
# 2. ATTENTION
# ============================================================

class Attention(nn.Module):

    def __init__(
        self,
        hidden_dim
    ):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.v = nn.Linear(
            hidden_dim,
            1,
            bias=False
        )

    def forward(
        self,
        hidden,
        encoder_outputs
    ):

        # ----------------------------------------------------
        # hidden:
        # [num_layers, batch_size, hidden_dim]
        #
        # We use the final decoder/encoder hidden state.
        # ----------------------------------------------------

        batch_size = (
            encoder_outputs.shape[0]
        )

        source_length = (
            encoder_outputs.shape[1]
        )

        # Last hidden state
        hidden = hidden[-1]

        # [batch_size, source_length, hidden_dim]

        hidden = hidden.unsqueeze(1)

        hidden = hidden.repeat(
            1,
            source_length,
            1
        )

        # ----------------------------------------------------
        # Combine hidden state with encoder outputs
        # ----------------------------------------------------

        energy = torch.tanh(
            self.attention(
                torch.cat(
                    (
                        hidden,
                        encoder_outputs
                    ),
                    dim=2
                )
            )
        )

        # ----------------------------------------------------
        # Calculate attention scores
        # ----------------------------------------------------

        attention = self.v(
            energy
        ).squeeze(2)

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        return torch.softmax(
            attention,
            dim=1
        )


# ============================================================
# 3. DECODER
# ============================================================

class Decoder(nn.Module):

    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        attention,
        num_layers=1,
        dropout=0.2
    ):

        super().__init__()

        self.output_dim = output_dim

        self.attention = attention

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.LSTM(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0
            )
        )

        self.fc_out = nn.Linear(
            hidden_dim * 2 + embedding_dim,
            output_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs
    ):

        # ----------------------------------------------------
        # input_token:
        # [batch_size]
        # ----------------------------------------------------

        input_token = input_token.unsqueeze(
            1
        )

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        embedded = self.dropout(
            self.embedding(input_token)
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        attention_weights = self.attention(
            hidden,
            encoder_outputs
        )

        # ----------------------------------------------------
        # Add dimension
        # ----------------------------------------------------

        attention_weights = attention_weights.unsqueeze(
            1
        )

        # ----------------------------------------------------
        # Context vector
        # ----------------------------------------------------

        context = torch.bmm(
            attention_weights,
            encoder_outputs
        )

        # context:
        # [batch_size, 1, hidden_dim]

        # ----------------------------------------------------
        # Decoder input
        # ----------------------------------------------------

        rnn_input = torch.cat(
            (
                embedded,
                context
            ),
            dim=2
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        output, (hidden, cell) = self.rnn(
            rnn_input,
            (hidden, cell)
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = self.fc_out(
            torch.cat(
                (
                    output,
                    context,
                    embedded
                ),
                dim=2
            )
        )

        # ----------------------------------------------------
        # Remove sequence dimension
        # ----------------------------------------------------

        prediction = prediction.squeeze(
            1
        )

        return (
            prediction,
            hidden,
            cell,
            attention_weights.squeeze(1)
        )


# ============================================================
# 4. SEQ2SEQ MODEL
# ============================================================

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        source,
        target,
        teacher_forcing_ratio=0.5
    ):

        # ----------------------------------------------------
        # source:
        # [batch_size, source_length]
        #
        # target:
        # [batch_size, target_length]
        # ----------------------------------------------------

        batch_size = source.shape[0]

        target_length = target.shape[1]

        target_vocab_size = (
            self.decoder.output_dim
        )

        # ----------------------------------------------------
        # Store predictions
        # ----------------------------------------------------

        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=self.device
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoder_outputs, hidden, cell = (
            self.encoder(source)
        )

        # ----------------------------------------------------
        # First decoder input = <sos>
        # ----------------------------------------------------

        input_token = target[:, 0]

        # ----------------------------------------------------
        # Decoder loop
        # ----------------------------------------------------

        for t in range(
            1,
            target_length
        ):

            output, hidden, cell, _ = (
                self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )

            outputs[:, t] = output

            # ------------------------------------------------
            # Teacher forcing
            # ------------------------------------------------

            best_prediction = output.argmax(
                1
            )

            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            input_token = (
                target[:, t]
                if teacher_force
                else best_prediction
            )

        return outputs


# ============================================================
# 5. CREATE MODEL
# ============================================================

def create_model(
    source_vocab_size,
    target_vocab_size,
    device
):

    print("\n" + "=" * 70)
    print("CREATING NMT MODEL")
    print("=" * 70)

    print(
        f"Source vocabulary size: "
        f"{source_vocab_size}"
    )

    print(
        f"Target vocabulary size: "
        f"{target_vocab_size}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Hidden dimension: "
        f"{HIDDEN_DIM}"
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder = Encoder(
        input_dim=source_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    attention = Attention(
        HIDDEN_DIM
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder = Decoder(
        output_dim=target_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        attention=attention,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    # --------------------------------------------------------
    # Seq2Seq
    # --------------------------------------------------------

    model = Seq2Seq(
        encoder,
        decoder,
        device
    )

    # Xavier initialization
    def initialize_weights(m):

        for name, parameter in m.named_parameters():

            if "weight" in name:

                nn.init.xavier_uniform_(
                    parameter
                )

            elif "bias" in name:

                nn.init.constant_(
                    parameter,
                    0
                )

    model.apply(
        initialize_weights
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Number of parameters
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        "\nModel created successfully."
    )

    return model


# ============================================================
# 6. TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING NMT MODEL")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # Dummy vocabulary sizes
    source_vocab_size = 10000
    target_vocab_size = 12000

    # Create model
    model = create_model(
        source_vocab_size,
        target_vocab_size,
        device
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    batch_size = 4
    source_length = 30
    target_length = 30

    source = torch.randint(
        0,
        source_vocab_size,
        (
            batch_size,
            source_length
        )
    ).to(device)

    target = torch.randint(
        0,
        target_vocab_size,
        (
            batch_size,
            target_length
        )
    ).to(device)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    print(
        "\nRunning test forward pass..."
    )

    output = model(
        source,
        target,
        teacher_forcing_ratio=0.5
    )

    print(
        f"Input shape: "
        f"{source.shape}"
    )

    print(
        f"Target shape: "
        f"{target.shape}"
    )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        "\nModel test completed successfully."
    )
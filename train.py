# ============================================================
# PART 3: NMT TRAINING
# English -> Hindi
# Encoder-Decoder LSTM with Attention
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"

TRAIN_FILE = os.path.join(
    DATA_DIR,
    "train.csv"
)

VAL_FILE = os.path.join(
    DATA_DIR,
    "validation.csv"
)

SOURCE_VOCAB_FILE = os.path.join(
    ARTIFACTS_DIR,
    "source_vocab.json"
)

TARGET_VOCAB_FILE = os.path.join(
    ARTIFACTS_DIR,
    "target_vocab.json"
)

MODEL_FILE = os.path.join(
    ARTIFACTS_DIR,
    "best_model.pt"
)

HISTORY_FILE = os.path.join(
    ARTIFACTS_DIR,
    "training_history.json"
)


# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 10

LEARNING_RATE = 0.0005

CLIP = 1.0

TEACHER_FORCING_RATIO = 0.5

MAX_SOURCE_LENGTH = 30

MAX_TARGET_LENGTH = 30

RANDOM_SEED = 42


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SET RANDOM SEEDS
# ============================================================

def set_seed():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


# ============================================================
# 1. LOAD VOCABULARY
# ============================================================

def load_vocabulary(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(
            file
        )

    return vocabulary


# ============================================================
# 2. TOKENIZATION
# ============================================================

def tokenize_english(text):

    import re

    text = str(text).lower()

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


def tokenize_hindi(text):

    import re

    text = str(text)

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


# ============================================================
# 3. NUMERICALIZE
# ============================================================

def numericalize(
    tokens,
    vocabulary
):

    unk_index = vocabulary[
        UNK_TOKEN
    ]

    return [
        vocabulary.get(
            token,
            unk_index
        )
        for token in tokens
    ]


# ============================================================
# 4. PAD / TRUNCATE
# ============================================================

def pad_sequence(
    sequence,
    max_length,
    pad_index,
    eos_index
):

    # --------------------------------------------------------
    # Truncate
    # --------------------------------------------------------

    if len(sequence) > max_length:

        sequence = sequence[
            :max_length
        ]

        # Ensure EOS at the end
        sequence[-1] = eos_index

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    while len(sequence) < max_length:

        sequence.append(
            pad_index
        )

    return sequence


# ============================================================
# 5. PROCESS SENTENCE
# ============================================================

def process_source(
    text,
    vocabulary
):

    tokens = tokenize_english(
        text
    )

    tokens = [
        SOS_TOKEN
    ] + tokens + [
        EOS_TOKEN
    ]

    ids = numericalize(
        tokens,
        vocabulary
    )

    ids = pad_sequence(
        ids,
        MAX_SOURCE_LENGTH,
        vocabulary[PAD_TOKEN],
        vocabulary[EOS_TOKEN]
    )

    return ids


def process_target(
    text,
    vocabulary
):

    tokens = tokenize_hindi(
        text
    )

    tokens = [
        SOS_TOKEN
    ] + tokens + [
        EOS_TOKEN
    ]

    ids = numericalize(
        tokens,
        vocabulary
    )

    ids = pad_sequence(
        ids,
        MAX_TARGET_LENGTH,
        vocabulary[PAD_TOKEN],
        vocabulary[EOS_TOKEN]
    )

    return ids


# ============================================================
# 6. DATASET CLASS
# ============================================================

class TranslationDataset(Dataset):

    def __init__(
        self,
        dataframe,
        source_vocab,
        target_vocab
    ):

        self.dataframe = dataframe

        self.source_vocab = source_vocab

        self.target_vocab = target_vocab

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(self, index):

        row = self.dataframe.iloc[
            index
        ]

        source = process_source(
            row["English"],
            self.source_vocab
        )

        target = process_target(
            row["Hindi"],
            self.target_vocab
        )

        return (
            torch.tensor(
                source,
                dtype=torch.long
            ),
            torch.tensor(
                target,
                dtype=torch.long
            )
        )


# ============================================================
# 7. LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    train_df = pd.read_csv(
        TRAIN_FILE,
        encoding="utf-8-sig"
    )

    val_df = pd.read_csv(
        VAL_FILE,
        encoding="utf-8-sig"
    )

    print(
        f"Training samples   : {len(train_df)}"
    )

    print(
        f"Validation samples : {len(val_df)}"
    )

    return (
        train_df,
        val_df
    )


# ============================================================
# 8. CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    train_df,
    val_df,
    source_vocab,
    target_vocab
):

    train_dataset = TranslationDataset(
        train_df,
        source_vocab,
        target_vocab
    )

    val_dataset = TranslationDataset(
        val_df,
        source_vocab,
        target_vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# 9. TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion
):

    model.train()

    epoch_loss = 0

    for batch_index, (
        source,
        target
    ) in enumerate(loader):

        source = source.to(
            DEVICE
        )

        target = target.to(
            DEVICE
        )

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(
            source,
            target,
            teacher_forcing_ratio=
                TEACHER_FORCING_RATIO
        )

        # ----------------------------------------------------
        # Ignore first token <sos>
        # ----------------------------------------------------

        output_dim = (
            output.shape[-1]
        )

        output = output[
            :, 1:
        ].reshape(
            -1,
            output_dim
        )

        target = target[
            :, 1:
        ].reshape(
            -1
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            target
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            CLIP
        )

        optimizer.step()

        epoch_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 100 == 0:

            print(
                f"Batch "
                f"{batch_index + 1}/"
                f"{len(loader)} "
                f"Loss: "
                f"{loss.item():.4f}"
            )

    return (
        epoch_loss /
        len(loader)
    )


# ============================================================
# 10. VALIDATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for source, target in loader:

            source = source.to(
                DEVICE
            )

            target = target.to(
                DEVICE
            )

            output = model(
                source,
                target,
                teacher_forcing_ratio=0
            )

            output_dim = (
                output.shape[-1]
            )

            output = output[
                :, 1:
            ].reshape(
                -1,
                output_dim
            )

            target = target[
                :, 1:
            ].reshape(
                -1
            )

            loss = criterion(
                output,
                target
            )

            epoch_loss += (
                loss.item()
            )

    return (
        epoch_loss /
        len(loader)
    )


# ============================================================
# 11. MAIN TRAINING
# ============================================================

def main():

    set_seed()

    os.makedirs(
        ARTIFACTS_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("ENGLISH -> HINDI NMT TRAINING")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load vocabularies
    # --------------------------------------------------------

    print("\nLoading vocabularies...")

    source_vocab = load_vocabulary(
        SOURCE_VOCAB_FILE
    )

    target_vocab = load_vocabulary(
        TARGET_VOCAB_FILE
    )

    print(
        f"Source vocabulary: "
        f"{len(source_vocab)}"
    )

    print(
        f"Target vocabulary: "
        f"{len(target_vocab)}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, val_df = load_data()

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader, val_loader = (
        create_dataloaders(
            train_df,
            val_df,
            source_vocab,
            target_vocab
        )
    )

    print(
        f"\nTraining batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model(
        source_vocab_size=len(
            source_vocab
        ),
        target_vocab_size=len(
            target_vocab
        ),
        device=DEVICE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=
        target_vocab[PAD_TOKEN]
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    train_losses = []

    val_losses = []

    best_validation_loss = float(
        "inf"
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        NUM_EPOCHS
    ):

        print("\n" + "=" * 70)

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = evaluate(
            model,
            val_loader,
            criterion
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        print(
            f"\nTraining Loss   : "
            f"{train_loss:.4f}"
        )

        print(
            f"Validation Loss : "
            f"{val_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_validation_loss:

            best_validation_loss = (
                val_loss
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "source_vocab_size":
                        len(source_vocab),

                    "target_vocab_size":
                        len(target_vocab),

                    "embedding_dim":
                        256,

                    "hidden_dim":
                        512,

                    "num_layers":
                        1,

                    "dropout":
                        0.2,

                    "max_source_length":
                        MAX_SOURCE_LENGTH,

                    "max_target_length":
                        MAX_TARGET_LENGTH
                },
                MODEL_FILE
            )

            print(
                "\nBest model saved!"
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history = {

        "train_loss":
            train_losses,

        "validation_loss":
            val_losses,

        "epochs":
            NUM_EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "teacher_forcing_ratio":
            TEACHER_FORCING_RATIO
    }

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"\nBest validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"\nModel saved at:"
        f"\n{MODEL_FILE}"
    )

    print(
        f"\nTraining history saved at:"
        f"\n{HISTORY_FILE}"
    )

    print("\n" + "=" * 70)
    print("NEXT STEP: GENERATE LOSS CURVE")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
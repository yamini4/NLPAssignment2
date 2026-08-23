import os
import re
import json
import torch
import streamlit as st

from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

ARTIFACTS_DIR = "artifacts"

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

MAX_SOURCE_LENGTH = 30
MAX_TARGET_LENGTH = 30

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="English to Hindi Translator",
    page_icon="🇮🇳",
    layout="centered"
)


# ============================================================
# LOAD VOCABULARY
# ============================================================

@st.cache_resource
def load_vocabulary(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_nmt_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    source_vocab = load_vocabulary(
        SOURCE_VOCAB_FILE
    )

    target_vocab = load_vocabulary(
        TARGET_VOCAB_FILE
    )

    model = create_model(
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        device=device
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return (
        model,
        source_vocab,
        target_vocab,
        device
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_english(text):

    text = str(text).lower()

    tokens = re.findall(
        r"\w+|[^\w\s]",
        text,
        flags=re.UNICODE
    )

    return tokens


# ============================================================
# NUMERICALIZE
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
# PAD / TRUNCATE
# ============================================================

def pad_sequence(
    sequence,
    max_length,
    pad_index,
    eos_index
):

    sequence = list(sequence)

    if len(sequence) > max_length:

        sequence = sequence[
            :max_length
        ]

        sequence[-1] = eos_index

    while len(sequence) < max_length:

        sequence.append(
            pad_index
        )

    return sequence


# ============================================================
# PROCESS ENGLISH
# ============================================================

def process_source(
    text,
    source_vocab
):

    tokens = tokenize_english(
        text
    )

    tokens = (
        [SOS_TOKEN]
        + tokens
        + [EOS_TOKEN]
    )

    ids = numericalize(
        tokens,
        source_vocab
    )

    ids = pad_sequence(
        ids,
        MAX_SOURCE_LENGTH,
        source_vocab[PAD_TOKEN],
        source_vocab[EOS_TOKEN]
    )

    return ids


# ============================================================
# DETOKENIZE
# ============================================================

def detokenize(tokens):

    text = " ".join(tokens)

    text = re.sub(
        r"\s+([।,!?;:])",
        r"\1",
        text
    )

    text = re.sub(
        r"\s*/\s*",
        "/",
        text
    )

    return text.strip()


# ============================================================
# TRANSLATION
# ============================================================

def translate_sentence(
    sentence,
    model,
    source_vocab,
    target_vocab,
    device
):

    target_index_to_token = {
        int(index): token
        for token, index
        in target_vocab.items()
    }

    source_ids = process_source(
        sentence,
        source_vocab
    )

    source_tensor = torch.tensor(
        source_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    with torch.inference_mode():

        encoder_outputs, hidden, cell = (
            model.encoder(
                source_tensor
            )
        )

    # --------------------------------------------------------
    # Start decoder with SOS
    # --------------------------------------------------------

    input_token = torch.tensor(
        [
            target_vocab[SOS_TOKEN]
        ],
        dtype=torch.long,
        device=device
    )

    generated_tokens = []

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    for _ in range(
        MAX_TARGET_LENGTH - 1
    ):

        with torch.inference_mode():

            output, hidden, cell, _ = (
                model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs
                )
            )

        prediction = output.argmax(
            dim=1
        )

        prediction_id = prediction.item()

        # EOS
        if prediction_id == target_vocab[
            EOS_TOKEN
        ]:
            break

        # Ignore PAD
        if prediction_id != target_vocab[
            PAD_TOKEN
        ]:

            token = target_index_to_token.get(
                prediction_id,
                UNK_TOKEN
            )

            generated_tokens.append(
                token
            )

        input_token = prediction

    return detokenize(
        generated_tokens
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    (
        model,
        source_vocab,
        target_vocab,
        device
    ) = load_nmt_model()

except Exception as error:

    st.error(
        f"Could not load the trained model:\n\n{error}"
    )

    st.stop()


# ============================================================
# USER INTERFACE
# ============================================================

st.title("🇮🇳 English → Hindi Translator")

st.write(
    "Neural Machine Translation using "
    "Encoder-Decoder LSTM with Attention"
)

st.divider()


# ============================================================
# INPUT
# ============================================================

english_text = st.text_area(
    "Enter English text:",
    height=150,
    placeholder="Example: The government is working for the people."
)


# ============================================================
# TRANSLATE BUTTON
# ============================================================

if st.button(
    "🔄 Translate to Hindi",
    type="primary"
):

    if not english_text.strip():

        st.warning(
            "Please enter an English sentence."
        )

    else:

        with st.spinner(
            "Translating..."
        ):

            try:

                hindi_translation = (
                    translate_sentence(
                        english_text,
                        model,
                        source_vocab,
                        target_vocab,
                        device
                    )
                )

                st.subheader(
                    "Hindi Translation"
                )

                st.success(
                    hindi_translation
                )

            except Exception as error:

                st.error(
                    f"Translation failed: {error}"
                )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.caption(
    "Model: Encoder-Decoder LSTM with Attention"
)

st.caption(
    "Translation direction: English → Hindi"
)

st.caption(
    f"Device: {device}"
)
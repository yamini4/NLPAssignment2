# English to Hindi Neural Machine Translation

## Group 109

This project implements an **English-to-Hindi Neural Machine Translation (NMT)** system using an **Encoder-Decoder LSTM architecture with Attention**.

The trained model translates English sentences into Hindi through a Streamlit web application.

---

## 1. Project Overview

The objective of this project is to develop an English-to-Hindi machine translation system using deep learning.

The system consists of the following major components:

1. Dataset preparation
2. Text preprocessing and normalization
3. English and Hindi tokenization
4. Source and target vocabulary creation
5. Numericalization and padding
6. Encoder-Decoder LSTM model
7. Attention mechanism
8. Model training
9. Model evaluation
10. Streamlit-based web application

The application accepts an English sentence from the user and generates the corresponding Hindi translation.

---

## 2. Problem Statement

Develop an NLP-based Neural Machine Translation system that translates sentences from English to Hindi.

The system should:

- Accept English text as input.
- Process and tokenize the input.
- Convert words into vocabulary indices.
- Pass the input through an Encoder-Decoder neural network.
- Use an Attention mechanism during decoding.
- Generate Hindi output.
- Display the translated sentence through a web application.

---

## 3. Dataset

### Dataset Name

IIT Bombay English-Hindi Parallel Corpus

### Dataset Source

The dataset was obtained from the publicly available Hugging Face dataset:

`cfilt/iitb-english-hindi`

### Language Pair

English → Hindi

### Dataset Size

Total sentence pairs used:

**47,156**

Dataset split:

| Dataset | Number of Samples |
|---|---:|
| Training | 37,724 |
| Validation | 4,716 |
| Testing | 4,716 |
| Total | 47,156 |

### License

The dataset is publicly available through the IIT Bombay English-Hindi corpus. 
The dataset's original licensing and usage conditions should be followed when redistributing or using the dataset.

The project does not redistribute the complete original dataset.

---

## 4. Technology Stack

The project was implemented using:

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib

### Python Version

Recommended:

`Python 3.10` or `Python 3.11`

---

## 5. Model Architecture

The project uses an Encoder-Decoder architecture based on LSTM networks.

### Encoder

The Encoder processes the English input sentence and converts it into hidden representations.

The Encoder contains:

- Word embedding layer
- LSTM layer
- Hidden state
- Cell state

### Attention

An Attention mechanism is used by the Decoder to focus on relevant parts of the encoded English sentence while generating each Hindi word.

This helps the decoder use different parts of the input sentence at different decoding steps.

### Decoder

The Decoder generates the Hindi translation sequentially.

At every decoding step:

1. The previous Hindi token is provided to the Decoder.
2. Attention weights are calculated.
3. Encoder outputs are combined using the attention weights.
4. The Decoder predicts the next Hindi token.
5. The process continues until `<eos>` is generated.

---

## 6. Special Tokens

The following special tokens are used:

| Token | ID | Purpose |
|---|---:|---|
| `<pad>` | 0 | Padding |
| `<unk>` | 1 | Unknown word |
| `<sos>` | 2 | Start of sentence |
| `<eos>` | 3 | End of sentence |

---

## 7. Preprocessing

The preprocessing pipeline performs the following operations:

### 7.1 Unicode Normalization

Unicode normalization is performed using NFC normalization.

This helps maintain consistency in Hindi Unicode characters.

### 7.2 Text Cleaning

The following preprocessing operations are performed:

- Removal of unnecessary spaces
- Removal of leading and trailing spaces
- Unicode normalization
- Lowercasing of English text

### 7.3 English Tokenization

English sentences are separated into:

- Words
- Numbers
- Punctuation

Example:

```text
How are you?


----------------
Sample Test Data
----------------
Sample English input files are available in:

sample_data/sample_input.txt
sample_data/sample_input_2.txt

These files can be used to test the English-to-Hindi translation application.
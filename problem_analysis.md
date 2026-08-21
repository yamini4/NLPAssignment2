# Part 1: Problem Analysis

## 1.1 Application Domain

The proposed application belongs to the Natural Language Processing (NLP)
and Neural Machine Translation (NMT) domain. The system is designed to
automatically translate English content into an Indian regional language.
The application can be useful for digital content, educational material,
government notices, product documentation, and other English-language
resources.

## 1.2 Target Users

The target users of the application include:

- Students and learners
- Regional-language users
- Government-service users
- Users who have difficulty understanding English content
- Content creators
- Organizations requiring bilingual content
- Users accessing educational and technical documentation

## 1.3 Problem Statement

Digital content, product documentation, government notices, and learning
materials are largely available in English. This creates an accessibility
barrier for users who prefer Indian regional languages. Manual translation
is time-consuming, expensive, and may result in inconsistent terminology.

The objective of this project is to develop an end-to-end Neural Machine
Translation application that automatically translates English sentences
into Hindi using an Encoder-Decoder architecture. The system learns
translation patterns from an aligned English-Hindi parallel corpus and
generates Hindi translations for previously unseen English input.

## 1.4 Functional Requirements

The application shall:

1. Accept English text as input.
2. Translate English text into Hindi.
3. Display the original English sentence and generated Hindi translation.
4. Support text-based translation through a web interface.
5. Support `.txt` file upload for batch translation.
6. Support `.csv` file upload containing English sentences.
7. Perform text cleaning and Unicode normalization.
8. Tokenize source and target sentences.
9. Use separate source and target vocabularies.
10. Add start-of-sequence and end-of-sequence tokens.
11. Apply padding and truncation to fixed sequence lengths.
12. Generate translations using an Encoder-Decoder NMT model.
13. Use an attention mechanism to improve translation quality.
14. Evaluate the trained model using suitable metrics such as BLEU and chrF.
15. Display translation results through a Streamlit web application.

## 1.5 Input Specification

The source language is English.

The application accepts:

- English sentences entered manually.
- English text uploaded through a `.txt` file.
- English sentences contained in a `.csv` file.

Example input:

> How are you?

## 1.6 Output Specification

The target language is Hindi.

The system generates the corresponding Hindi translation.

Example:

**English:**

> How are you?

**Hindi:**

> நீங்கள் எப்படி இருக்கிறீர்கள்?

## 1.7 Language Pair

| Component | Specification |
|---|---|
| Source Language | English |
| Target Language | Hindi |
| Translation Direction | English → Hindi |
| Model Type | Neural Machine Translation |
| Architecture | Encoder-Decoder |
| Encoder | LSTM |
| Decoder | LSTM |
| Attention | Yes |
| Interface | Streamlit |
| Batch Input | `.txt`, `.csv` |
| Evaluation | BLEU, chrF |

## 1.8 Expected System Workflow

The overall system follows the workflow:

English Input
→ Text Cleaning
→ Tokenization
→ Source Vocabulary
→ Encoder
→ Attention Mechanism
→ Decoder
→ Target Vocabulary
→ Hindi Translation

For batch translation:

File Upload
→ Read English Sentences
→ Preprocessing
→ NMT Model
→ Hindi Translations
→ Display Results
→ Download Results

## 1.9 Expected Outcome

The expected outcome is a functional web-based NMT application capable of
translating English sentences into Hindi. The system should provide
reasonable translations for common sentences and demonstrate the ability
to handle short, medium, and relatively complex sentences. Translation
quality will be evaluated using automatic metrics and qualitative
observations.
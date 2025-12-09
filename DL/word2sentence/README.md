# Word-to-Sentence Module

A sentence formation module that converts recognized ASL words into natural, grammatically correct sentences using advanced language models.

## 🎯 Overview

The `word2sentence` module is a **critical component** of the Gesture2Globe application that transforms individual recognized ASL words into coherent, natural sentences. This module uses state-of-the-art language models to ensure grammatical correctness and natural flow.

## 🔗 Integration with Gesture2Globe

### How This Module is Used

The `word2sentence` module is automatically invoked by the backend server (`video-integration/backend/app.py`) during the ASL recognition pipeline:

1. **After Recognition**: When ASL signs are recognized, individual words are collected
2. **Sentence Formation**: This module converts the word sequence into a natural sentence
3. **Optimization**: If only one word is recognized, it's returned directly (no API call needed)
4. **Translation Ready**: The formed sentence is then passed to the translation module

### Integration Flow

```
ASL Recognition → Word Collection → Sentence Formation → Translation → TTS
```

The backend calls the sentence formation function:
```python
# In video-integration/backend/app.py
from sentence_former import best_sentence

# Form sentence from recognized words
sentence = _form_sentence(words)
```

### Importance

- **Natural Language**: Converts raw ASL glosses into readable, natural sentences
- **User Experience**: Makes recognition results more understandable and professional
- **Grammar Correction**: Ensures proper grammar and sentence structure
- **Context Awareness**: Uses language models to understand word relationships
- **Performance Optimization**: Skips processing for single-word cases

## 🏗️ Architecture

### Core Components

1. **FLAN-T5 Model**: Used for sentence generation
   - Generates multiple candidate sentences from input words
   - Ensures grammatical correctness
   - Maintains word order and meaning

2. **GPT-2 Model**: Used for fluency scoring
   - Scores candidate sentences for naturalness
   - Selects the most fluent and natural sentence
   - Ensures high-quality output

3. **Word Deduplication**: Removes repeated words while preserving order
   - Handles cases where the same sign is recognized multiple times
   - Maintains temporal order of recognized words

4. **Single Word Optimization**: 
   - If only one word is recognized, returns it directly
   - Avoids unnecessary API calls and processing
   - Improves performance for single-sign recognition

### Processing Pipeline

```
Input Words → Deduplication → Candidate Generation (FLAN-T5) → Fluency Scoring (GPT-2) → Best Sentence
```

### Key Features

- **Word Order Preservation**: Maintains the order in which signs were recognized
- **Deduplication**: Removes repeated words intelligently
- **Multiple Candidates**: Generates several sentence options and selects the best
- **Target Length Optimization**: Calculates optimal sentence length based on word count
- **Error Handling**: Graceful fallback to simple word joining if models fail

## 📁 Module Structure

```
word2sentence/
├── README.md              # This file
├── sentence_former.py     # Main sentence formation logic
├── requirements.txt       # Python dependencies
└── __init__.py           # Package initialization
```

## 🔧 Usage

### Direct Usage

```python
from sentence_former import best_sentence

# Input: List of recognized words
words = ['hello', 'how', 'are', 'you']

# Generate best sentence
sentence, candidates = best_sentence(words, n=6, target_len=9)
print(sentence)  # Output: "Hello, how are you?"
```

### Integration in Backend

The module is used in `video-integration/backend/app.py`:

```python
def _form_sentence(words):
    # Filter and deduplicate words
    unique_words = deduplicate(words)
    
    # Single word optimization
    if len(unique_words) == 1:
        return unique_words[0]
    
    # Generate sentence using word2sentence module
    sentence = best_sentence(unique_words, n=n, target_len=target_len)
    return sentence
```

## 📊 Performance

- **Single Word**: Instant return (no processing)
- **Multiple Words**: ~1-3 seconds depending on word count
- **Model Loading**: First call loads models (~10-30 seconds), subsequent calls are fast
- **Memory Usage**: ~2-4GB for loaded models
- **Quality**: High-quality, grammatically correct sentences

## 🛠️ Dependencies

Required Python packages (see `requirements.txt`):
- `torch` - PyTorch for model inference
- `transformers` - Hugging Face transformers library
- `nltk` - Natural Language Toolkit for text processing

## 🔄 Model Loading

The module uses lazy loading:
- Models are loaded on first use
- Models are cached in memory for subsequent calls
- NLTK data is downloaded automatically if missing

## 🐛 Troubleshooting

### Model Download Issues
- Ensure internet connection for first-time model downloads
- Models are downloaded from Hugging Face automatically
- Check disk space (models can be several GB)

### Memory Issues
- Models require significant RAM (~2-4GB)
- Consider using smaller models if memory is limited
- Close other applications to free memory

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that the module path is correct in backend imports
- Verify Python version (3.8+ required)

## 📝 Key Functions

### `best_sentence(words, n, target_len)`
Main function for sentence formation:
- **words**: List of input words
- **n**: Number of candidate sentences to generate
- **target_len**: Target sentence length
- **Returns**: Best sentence and top candidates

### Internal Functions
- `_ensure_models_loaded()`: Loads FLAN-T5 and GPT-2 models
- `_ensure_nltk_data()`: Downloads required NLTK data
- Word deduplication and normalization logic

## 🎓 Design Decisions

1. **Lazy Loading**: Models loaded only when needed to reduce startup time
2. **Single Word Optimization**: Direct return for single words improves performance
3. **Candidate Generation**: Multiple candidates ensure quality output
4. **Fluency Scoring**: GPT-2 scoring selects most natural sentence
5. **Error Handling**: Graceful fallback ensures system always returns a result

## 📚 References

- **FLAN-T5**: Google's instruction-tuned T5 model for text generation
- **GPT-2**: OpenAI's language model for fluency scoring
- **NLTK**: Natural Language Toolkit for text processing

---

**This module is essential for converting raw ASL recognition results into natural, readable sentences that users can understand and use.**


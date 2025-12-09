"""
Word to Sentence Formation Module
Uses FLAN-T5 for sentence generation and GPT-2 for fluency scoring.
Based on word2sentence.py implementation with enhanced word order preservation.
"""

import re
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GPT2TokenizerFast, GPT2LMHeadModel
import nltk
from nltk.stem import WordNetLemmatizer
from typing import Optional

# Module-level variables for models and utilities
_lemm = None
_device = None
_gen_tok = None
_gen_mod = None
_sc_tok = None
_sc_mod = None
_models_loaded = False

# Cache for lemmatization to avoid redundant calculations
_lemmatization_cache = {}

def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download("wordnet", quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

def _ensure_models_loaded():
    """Load and initialize all models. Safe to call multiple times."""
    global _lemm, _device, _gen_tok, _gen_mod, _sc_tok, _sc_mod, _models_loaded
    
    if _models_loaded:
        return
    
    try:
        print("[SENTENCE] Downloading NLTK data...")
        _ensure_nltk_data()
        
        _lemm = WordNetLemmatizer()
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[SENTENCE] Using device: {_device}")
        print("[SENTENCE] Loading FLAN-T5 generator model...")
        
        gen_name = "google/flan-t5-large"  # Use "google/flan-t5-base" if you have memory issues
        try:
            _gen_tok = AutoTokenizer.from_pretrained(gen_name)
            _gen_mod = AutoModelForSeq2SeqLM.from_pretrained(gen_name).to(_device).eval()
            print("[SENTENCE] FLAN-T5 model loaded successfully")
        except Exception as e:
            print(f"[SENTENCE ERROR] Failed to load FLAN-T5 model: {e}")
            print("[SENTENCE] Will use fallback sentence formation (simple concatenation)")
            _gen_tok = None
            _gen_mod = None
        
        print("[SENTENCE] Loading GPT-2 fluency scorer...")
        try:
            _sc_tok = GPT2TokenizerFast.from_pretrained("gpt2")
            _sc_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(_device).eval()
            print("[SENTENCE] GPT-2 model loaded successfully")
        except Exception as e:
            print(f"[SENTENCE ERROR] Failed to load GPT-2 model: {e}")
            print("[SENTENCE] Will continue without fluency scoring")
            _sc_tok = None
            _sc_mod = None
        
        _models_loaded = True
        
        if _gen_tok is not None and _gen_mod is not None:
            print("[SENTENCE] Models loaded successfully!\n")
        else:
            print("[SENTENCE WARNING] Models not fully loaded. Sentence formation will use fallback.\n")
    except Exception as e:
        print(f"[SENTENCE ERROR] Critical error during model loading: {e}")
        import traceback
        traceback.print_exc()
        _models_loaded = True  # Mark as loaded to prevent retry loops
        _gen_tok = None
        _gen_mod = None
        _sc_tok = None
        _sc_mod = None

def normalize_words(words):
    """Clean and normalize input words."""
    if not words:
        return []
    return [w.strip().lower() for w in words if w.strip()]

def lemmas_in_sentence(sentence):
    """Extract all lemmas from a sentence."""
    if not _models_loaded:
        _ensure_models_loaded()
    
    # Check cache first
    if sentence in _lemmatization_cache:
        return _lemmatization_cache[sentence]
    
    tokens = re.findall(r"[A-Za-z']+", sentence.lower())
    lemmas = set()
    for t in tokens:
        lemmas.add(_lemm.lemmatize(t, 'v'))
        lemmas.add(_lemm.lemmatize(t, 'n'))
        lemmas.add(_lemm.lemmatize(t, 'a'))
        lemmas.add(t)  # include original form
    
    # Cache the result
    _lemmatization_cache[sentence] = lemmas
    return lemmas

def word_count(s: str) -> int:
    """Count words in a sentence."""
    return len(re.findall(r"[A-Za-z']+", s))

def covers_all(sentence, required_words):
    """Check if sentence contains all required words (via lemmatization)."""
    if not _models_loaded:
        _ensure_models_loaded()
    sent_lemmas = lemmas_in_sentence(sentence)
    for word in required_words:
        word_lemmas = {
            _lemm.lemmatize(word, 'v'),
            _lemm.lemmatize(word, 'n'),
            _lemm.lemmatize(word, 'a'),
            word
        }
        if not word_lemmas & sent_lemmas:
            return False
    return True

def build_prompt(words):
    """Create a simple, concise prompt for short grammatical sentences with word order."""
    word_list = ", ".join(f'"{w}"' for w in words)
    target_min = len(words) + 2
    target_max = max(8, 2 * len(words) + 2)

    # Create ordered word list for emphasis
    ordered_list = " → ".join(f'"{w}"' for w in words)

    return f"""Task: Write ONE SHORT, SIMPLE sentence using ALL these words IN THIS ORDER: {ordered_list}

CRITICAL RULES:
1. Use EVERY word from the list EXACTLY ONCE (can change tense/form: "run"→"running", "book"→"books")
2. Keep the words IN THE GIVEN ORDER ONLY: {ordered_list}
3. Keep it SHORT and SIMPLE: {target_min}-{target_max} words total
4. Use basic fillers ONLY when needed: "the", "a", "and", "is", "was", "to", "in"
5. NO extra details, NO hallucinations, NO repetition
6. Must be grammatically correct

Good examples (SHORT, SIMPLE, and IN ORDER):
- Words: "cards" → "shuffle" → "Shuffle the cards."
- Words: "student" → "read" → "book" → "The student read a book."
- Words: "teacher" → "help" → "student" → "The teacher helped the student."
- Words: "dog" → "run" → "The dog is running."

BAD examples:
- ✗ Words: "teacher" → "help" → "student" but wrote "The student helped the teacher." (WRONG ORDER)
- ✗ "The dog is running in the park." (added "park")

Your SHORT sentence (keeping word order):"""

@torch.no_grad()
def perplexity(sentence: str) -> float:
    """Calculate perplexity score for fluency."""
    if not _models_loaded:
        _ensure_models_loaded()
    
    if _sc_tok is None or _sc_mod is None:
        # Return a default perplexity if GPT-2 isn't available
        return 50.0
    
    try:
        ids = _sc_tok.encode(sentence, return_tensors="pt").to(_device)
        out = _sc_mod(ids, labels=ids)
        return math.exp(out.loss.item())
    except Exception as e:
        print(f"[SENTENCE] Perplexity calculation error: {e}")
        return 1000.0

def word_order_score(sentence, required_words):
    """
    Calculate how well the sentence follows the required word order.
    Returns a score between 0 and 1, where 1 means perfect order.
    """
    if not _models_loaded:
        _ensure_models_loaded()
    sent_lemmas = lemmas_in_sentence(sentence)
    tokens = re.findall(r"[A-Za-z']+", sentence.lower())

    # Pre-compute word lemmas for all required words (cache-friendly)
    word_lemmas_list = []
    for word in required_words:
        # Check cache for word lemmatization
        cache_key = f"_word_{word}"
        if cache_key not in _lemmatization_cache:
            _lemmatization_cache[cache_key] = {
                _lemm.lemmatize(word, 'v'),
                _lemm.lemmatize(word, 'n'),
                _lemm.lemmatize(word, 'a'),
                word
            }
        word_lemmas_list.append(_lemmatization_cache[cache_key])

    # Find positions of each required word in the sentence
    positions = []
    for word_lemmas in word_lemmas_list:
        # Find the first occurrence of this word in the sentence
        found_pos = -1
        for i, token in enumerate(tokens):
            # Check cache for token lemmatization
            token_cache_key = f"_token_{token}"
            if token_cache_key not in _lemmatization_cache:
                _lemmatization_cache[token_cache_key] = {
                    _lemm.lemmatize(token, 'v'),
                    _lemm.lemmatize(token, 'n'),
                    _lemm.lemmatize(token, 'a'),
                    token
                }
            token_lemmas = _lemmatization_cache[token_cache_key]
            if word_lemmas & token_lemmas:
                found_pos = i
                break
        positions.append(found_pos)

    # Check if all words were found
    if -1 in positions:
        return 0.0

    # Check if positions are in increasing order
    is_ordered = all(positions[i] < positions[i+1] for i in range(len(positions)-1))

    if is_ordered:
        return 1.0
    else:
        # Calculate partial score based on how many pairs are in order
        correct_pairs = sum(1 for i in range(len(positions)-1) if positions[i] < positions[i+1])
        total_pairs = len(positions) - 1
        return correct_pairs / max(1, total_pairs) if total_pairs > 0 else 0.0

def coverage_score(sentence, required_words):
    """Calculate what fraction of required words are covered."""
    if not _models_loaded:
        _ensure_models_loaded()
    sent_lemmas = lemmas_in_sentence(sentence)
    hits = 0
    for word in required_words:
        # Use cached word lemmas if available
        cache_key = f"_word_{word}"
        if cache_key in _lemmatization_cache:
            word_lemmas = _lemmatization_cache[cache_key]
        else:
            word_lemmas = {
                _lemm.lemmatize(word, 'v'),
                _lemm.lemmatize(word, 'n'),
                _lemm.lemmatize(word, 'a'),
                word
            }
            _lemmatization_cache[cache_key] = word_lemmas
        if word_lemmas & sent_lemmas:
            hits += 1
    return hits / max(1, len(required_words))

@torch.no_grad()
def generate_candidates(words, n=40, max_new_tokens=40, temperature=0.85, top_p=0.92,
                        target_len: Optional[int]=None, len_tolerance: int=3):
    """
    Generate multiple SHORT candidate sentences with minimal fillers.
    """
    if not _models_loaded:
        _ensure_models_loaded()
    
    if _gen_tok is None or _gen_mod is None:
        # Fallback: return simple joined sentence
        print("[SENTENCE WARNING] Models not loaded, using fallback concatenation")
        sentence = " ".join(words).capitalize() + "."
        return [sentence]
    
    prompt = build_prompt(words)

    if target_len is not None:
        lo, hi = max(3, target_len - len_tolerance), target_len + len_tolerance + 2
        prompt += f" Target: {lo}-{hi} words."

    try:
        enc = _gen_tok(prompt, return_tensors="pt").to(_device)
        out = _gen_mod.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            eos_token_id=_gen_tok.eos_token_id,
        )

        texts = [_gen_tok.decode(o, skip_special_tokens=True).split("Your SHORT sentence:")[-1].strip()
                 for o in out]

        # Clean and normalize
        clean = []
        seen = set()
        for t in texts:
            # Extract just the sentence part
            t = t.split('\n')[0].strip()  # take first line only
            t = re.sub(r"\s+", " ", t).strip()
            # Remove any leading quotes or punctuation
            t = re.sub(r'^["\'\-•→]', '', t).strip()
            t = re.sub(r'["\']$', '', t).strip()

            if not re.search(r"[.!?]$", t):
                t += "."

            # Skip duplicates and invalid sentences - prefer SHORTER
            if t not in seen and 3 <= word_count(t) <= 15:  # stricter upper limit
                clean.append(t)
                seen.add(t)

        # Filter by length - prefer shorter sentences
        if target_len is not None:
            lo, hi = max(3, target_len - len_tolerance), target_len + len_tolerance + 2
            filtered = [t for t in clean if lo <= word_count(t) <= hi]
            if filtered:
                clean = filtered

        return clean
    except Exception as e:
        print(f"[SENTENCE] Generation error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return simple joined sentence
        print("[SENTENCE] Using fallback concatenation due to generation error")
        sentence = " ".join(words).capitalize() + "."
        return [sentence]

def best_sentence(words, n=None, target_len=None, verbose=False, max_retries=3):
    """
    Generate the best SHORT sentence that includes all required words.

    Args:
        words: List of words to include
        n: Number of candidates to generate (default: 5 * len(words))
        target_len: Target sentence length in words
        verbose: Print debugging information
        max_retries: Number of retry attempts if no perfect match found

    Returns:
        (best_sentence, top_5_candidates_with_scores)
        where top_5_candidates is a list of tuples: [(sentence, perplexity), ...]
    """
    if not _models_loaded:
        _ensure_models_loaded()
    
    # Check if models are actually loaded
    if _gen_tok is None or _gen_mod is None:
        print("[SENTENCE WARNING] Language models not available. Using fallback sentence formation.")
        print("[SENTENCE] This may be due to:")
        print("[SENTENCE]   - Missing dependencies (torch, transformers)")
        print("[SENTENCE]   - Network issues (models need to be downloaded)")
        print("[SENTENCE]   - Memory constraints")
        print("[SENTENCE] Falling back to simple word concatenation.")
        # Return simple concatenated sentence
        words = normalize_words(words)
        if not words:
            return "Please provide one or more content words.", []
        sentence = " ".join(words).capitalize() + "."
        return sentence, [(sentence, 100.0)]  # Return with high perplexity to indicate fallback
    
    words = normalize_words(words)
    if not words:
        return "Please provide one or more content words.", []

    # Generate fewer candidates initially for faster processing
    if n is None:
        n = max(20, 4 * len(words))  # Reduced from 30, 5*len(words)

    # Set SHORT target length
    if target_len is None:
        target_len = len(words) + 3  # much shorter default

    req = set(words)
    all_candidates = []

    # Try multiple times with different temperatures (reduced retries)
    max_retries = min(max_retries, 2)  # Limit to 2 retries max
    for retry in range(max_retries):
        if verbose and retry > 0:
            print(f"\n[SENTENCE] Retry {retry}: Generating shorter sentences...")

        temp = 0.8 + (retry * 0.15)  # start lower for more focused output
        cands = generate_candidates(words, n=n, target_len=target_len,
                                    len_tolerance=3, temperature=temp)
        all_candidates.extend(cands)

        # Check if we have any perfect matches - early stopping
        perfect = [c for c in all_candidates if covers_all(c, req)]
        if len(perfect) >= 5:  # want at least 5 good options
            if verbose:
                print(f"[SENTENCE] Found {len(perfect)} perfect short matches!")
            break

    # Remove duplicates
    unique_cands = list(dict.fromkeys(all_candidates))

    if verbose:
        print(f"[SENTENCE] Total unique candidates: {len(unique_cands)}")

    # First filter: sentences that cover ALL words
    perfect = [c for c in unique_cands if covers_all(c, req)]

    if verbose:
        print(f"[SENTENCE] Perfect matches (all words): {len(perfect)}")

    # If no perfect matches, get best partial matches
    if not perfect:
        if not verbose:
            print(f"[SENTENCE] Warning: Could not find sentence with ALL words. Showing best partial matches.")
        # Score by coverage (only for top candidates to save time)
        scored_by_coverage = [(c, coverage_score(c, req)) for c in unique_cands]
        scored_by_coverage.sort(key=lambda x: x[1], reverse=True)

        # Take fewer top candidates for faster processing
        cutoff = min(15, max(8, len(scored_by_coverage) // 4))  # Reduced from 20, max(10, //3)
        good = [s for s, _ in scored_by_coverage[:cutoff]]
    else:
        good = perfect

    if not good:
        return "Could not generate valid sentence.", []

    # Pre-score by coverage and order (fast operations) to reduce perplexity calculations
    pre_scored = []
    for s in good:
        cov = coverage_score(s, req)
        order = word_order_score(s, words)
        wc = word_count(s)
        brevity_score = 1.0 / (wc + 1)
        # Pre-score without perplexity (expensive operation)
        pre_scored.append((s, cov, wc, brevity_score, order))
    
    # Sort by coverage and order first, then take top candidates for perplexity
    pre_scored.sort(key=lambda x: (-x[1], -x[4], -x[3]))
    
    # Only calculate perplexity for top candidates (major optimization)
    top_for_ppl = min(12, len(pre_scored))  # Calculate perplexity for top 12 only
    scored = []
    for i, (s, cov, wc, brevity_score, order) in enumerate(pre_scored):
        if i < top_for_ppl:
            ppl = perplexity(s)  # Expensive operation - only for top candidates
        else:
            ppl = 50.0  # Default reasonable perplexity for others
        scored.append((s, ppl, cov, wc, brevity_score, order))

    # Sort by: 1) coverage (desc), 2) word_order (desc), 3) brevity (desc), 4) perplexity (asc)
    scored.sort(key=lambda x: (-x[2], -x[5], -x[4], x[1]))

    if verbose:
        print("\n[SENTENCE] Top 5 candidates (shorter is better, correct order prioritized):")
        for i, (sent, ppl, cov, wc, _, order) in enumerate(scored[:5], 1):
            missing = []
            for w in req:
                if not covers_all(sent, {w}):
                    missing.append(w)
            miss_str = f" [Missing: {missing}]" if missing else ""
            order_str = "PASS" if order == 1.0 else f"{order*100:.0f}%"
            print(f"{i}. [{cov*100:.0f}% coverage, order:{order_str}, {wc}w, ppl={ppl:.1f}]{miss_str} {sent}")

    best = scored[0][0]
    # Return top 5 as list of tuples: (sentence, perplexity)
    top_5 = [(s, p) for s, p, _, _, _, _ in scored[:5]]

    return best, top_5

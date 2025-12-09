"""
Word2Sentence package for forming natural sentences from word lists.
"""

from .sentence_former import best_sentence, generate_candidates, perplexity

__all__ = ['best_sentence', 'generate_candidates', 'perplexity']



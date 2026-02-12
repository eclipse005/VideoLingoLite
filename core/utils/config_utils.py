import re
import math
import pandas as pd
from .config_manager import load_key, update_key, update_keys

# -----------------------
# Safe CSV reading with encoding detection
# -----------------------

def safe_read_csv(filepath, **kwargs):
    """
    Read CSV with automatic encoding detection.
    Tries common encodings in order.
    """
    # Remove 'encoding' from kwargs if present
    kwargs.pop('encoding', None)

    # List of encodings to try, in order of preference
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']

    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception:
            # If it's not a decoding error, something else is wrong - don't suppress
            raise

    # If all encodings fail, raise the original error
    raise UnicodeDecodeError(f"Could not read {filepath} with any common encoding")

        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

# -----------------------
# length checking utils
# -----------------------

def get_effective_length(text: str, lang: str) -> int:
    """
    Get effective length (excluding punctuation).
    For space-separated languages: count valid words
    For non-space languages: count valid characters
    """
    if lang in load_key('language_split_with_space'):
        # Space-separated languages: count only valid words (excluding pure punctuation)
        words = text.split()
        valid_words = [w for w in words if re.match(r'\w', w)]
        return len(valid_words)
    else:
        # Non-space languages: count only valid characters (excluding punctuation)
        valid_chars = [c for c in text if re.match(r'\w', c)]
        return len(valid_chars)

def get_language_length_limit(lang: str, limit_type: str = 'origin') -> int:
    """
    Get the soft length limit for a language.
    Args:
        lang: ISO language code (e.g., 'en', 'ja', 'zh')
        limit_type: 'origin' for original text, 'translate' for translation
    Returns:
        Soft limit value (int)
    """
    config_key = f'{limit_type}_length'
    limits = load_key(config_key)
    return limits.get(lang, limits.get('en', 15))  # Default to English limit or 15

def get_hard_limit(soft_limit: int, lang: str) -> int:
    """
    Calculate hard limit from soft limit using split_ratio.
    Args:
        soft_limit: The soft limit value
        lang: ISO language code
    Returns:
        Hard limit (ceiling of soft_limit × ratio)
    """
    split_ratios = load_key('split_ratio')
    ratio = split_ratios.get(lang, split_ratios.get('en', 1.4))
    return math.ceil(soft_limit * ratio)

def check_length_exceeds(text: str, limit: int, lang: str) -> bool:
    """
    Check if text exceeds the hard limit.
    Args:
        text: The text to check
        limit: The soft limit value
        lang: ISO language code
    Returns:
        True if text exceeds hard limit, False otherwise
    """
    actual_length = get_effective_length(text, lang)
    hard_limit = get_hard_limit(limit, lang)
    return actual_length > hard_limit

if __name__ == "__main__":
    print(load_key('language_split_with_space'))

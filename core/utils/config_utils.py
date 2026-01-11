from ruamel.yaml import YAML
import threading
import re
import math

CONFIG_PATH = 'config.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

# -----------------------
# load & update config
# -----------------------

def load_key(key):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{k}' not found in configuration")
    return value

def update_key(key, new_value):
    with lock:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")
        
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

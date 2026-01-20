from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """词/字级别的语音识别单元，携带时间戳"""
    text: str           # 词/字内容（如 "Hello", "world."）
    start: float        # 开始时间（秒）
    end: float          # 结束时间（秒）
    speaker_id: Optional[str] = None  # 说话人ID
    index: int = 0      # 在 cleaned_chunks.csv 中的行号

    @property
    def duration(self) -> float:
        """获取该词的时长"""
        return self.end - self.start


@dataclass
class Sentence:
    """句子级别的对象，由多个 Chunk 组成"""
    chunks: List[Chunk]     # 组成这句话的所有 Chunk
    text: str               # 从 chunks 拼接的完整文本
    start: float            # = chunks[0].start
    end: float              # = chunks[-1].end
    translation: str = ""   # 翻译文本
    index: int = 0          # 句子序号
    is_split: bool = False  # 是否被 LLM 切分过

    @property
    def duration(self) -> float:
        """获取这句话的时长"""
        return self.end - self.start

    def update_timestamps(self):
        """根据 chunks 更新 start 和 end 时间戳"""
        if self.chunks:
            self.start = self.chunks[0].start
            self.end = self.chunks[-1].end


# ------------------------------------------
# 定义中间产出文件
# ------------------------------------------

_2_CLEANED_CHUNKS = "output/log/cleaned_chunks.csv"
_3_1_SPLIT_BY_NLP = "output/log/split_by_nlp.txt"
_3_2_SPLIT_BY_MEANING_RAW = "output/log/split_by_meaning_raw.txt"  # LLM组句/Parakeet segments 原始结果
_3_2_SPLIT_BY_MEANING = "output/log/split_by_meaning.txt"  # 切分长句后的最终结果
_4_1_TERMINOLOGY = "output/log/terminology.json"
_4_2_TRANSLATION = "output/log/translation_results.csv"
_5_SPLIT_SUB = "output/log/translation_results_for_subtitles.csv"
_5_REMERGED = "output/log/translation_results_remerged.csv"



# ------------------------------------------
# 定义音频文件
# ------------------------------------------
_OUTPUT_DIR = "output"
_AUDIO_DIR = "output/audio"
_RAW_AUDIO_FILE = "output/audio/raw.mp3"
_VOCAL_AUDIO_FILE = "output/audio/vocals.wav"
_AUDIO_REFERS_DIR = "output/audio/refers"
_AUDIO_SEGS_DIR = "output/audio/segs"
_AUDIO_TMP_DIR = "output/audio/tmp"

# ------------------------------------------
# 导出
# ------------------------------------------

__all__ = [
    "Chunk",
    "Sentence",
    "_2_CLEANED_CHUNKS",
    "_3_1_SPLIT_BY_NLP",
    "_3_2_SPLIT_BY_MEANING_RAW",
    "_3_2_SPLIT_BY_MEANING",
    "_4_1_TERMINOLOGY",
    "_4_2_TRANSLATION",
    "_5_SPLIT_SUB",
    "_5_REMERGED",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR",
    "TARGET_LANG_MAP"
]


# ------------------------------------------
# 目标语言映射表（自然语言描述 → ISO 代码）
# ------------------------------------------
TARGET_LANG_MAP = {
    # 中文
    '简体中文': 'zh', '中文': 'zh', 'Chinese': 'zh', 'zh': 'zh', '汉语': 'zh',
    '繁体中文': 'zh', '正體中文': 'zh', '國語': 'zh',
    
    # 英语
    'English': 'en', '英语': 'en', '英文': 'en', 'en': 'en',
    
    # 日语
    '日本語': 'ja', '日语': 'ja', 'Japanese': 'ja', 'ja': 'ja',
    
    # 西班牙语
    'Español': 'es', '西班牙语': 'es', 'es': 'es', 'Spanish': 'es',
    
    # 法语
    'Français': 'fr', '法语': 'fr', 'fr': 'fr', 'French': 'fr',
    
    # 德语
    'Deutsch': 'de', '德语': 'de', 'de': 'de', 'German': 'de',
    
    # 意大利语
    'Italiano': 'it', '意大利语': 'it', 'it': 'it', 'Italian': 'it',
    
    # 俄语
    'Русский': 'ru', '俄语': 'ru', 'ru': 'ru', 'Russian': 'ru',
    
    # 韩语
    '한국어': 'ko', '韩语': 'ko', 'ko': 'ko', 'Korean': 'ko',
    
    # 葡萄牙语
    'Português': 'pt', '葡萄牙语': 'pt', 'pt': 'pt', 'Portuguese': 'pt',
    
    # 阿拉伯语
    'العربية': 'ar', '阿拉伯语': 'ar', 'ar': 'ar', 'Arabic': 'ar',
    
    # 荷兰语
    'Nederlands': 'nl', '荷兰语': 'nl', 'nl': 'nl', 'Dutch': 'nl',
    
    # 瑞典语
    'Svenska': 'sv', '瑞典语': 'sv', 'sv': 'sv', 'Swedish': 'sv',
    
    # 波兰语
    'Polski': 'pl', '波兰语': 'pl', 'pl': 'pl', 'Polish': 'pl',
    
    # 土耳其语
    'Türkçe': 'tr', '土耳其语': 'tr', 'tr': 'tr', 'Turkish': 'tr',
    
    # 泰语
    'ภาษาไทย': 'th', '泰语': 'th', 'th': 'th', 'Thai': 'th',
    
    # 越南语
    'Tiếng Việt': 'vi', '越南语': 'vi', 'vi': 'vi', 'Vietnamese': 'vi',
    
    # 印地语
    'हिन्दी': 'hi', '印地语': 'hi', 'hi': 'hi', 'Hindi': 'hi',
    
    # 希腊语
    'Ελληνικά': 'el', '希腊语': 'el', 'el': 'el', 'Greek': 'el',
    
    # 捷克语
    'Čeština': 'cs', '捷克语': 'cs', 'cs': 'cs', 'Czech': 'cs',
    
    # 匈牙利语
    'Magyar': 'hu', '匈牙利语': 'hu', 'hu': 'hu', 'Hungarian': 'hu',
    
    # 芬兰语
    'Suomi': 'fi', '芬兰语': 'fi', 'fi': 'fi', 'Finnish': 'fi',
    
    # 丹麦语
    'Dansk': 'da', '丹麦语': 'da', 'da': 'da', 'Danish': 'da',
    
    # 挪威语
    'Norsk': 'no', '挪威语': 'no', 'no': 'no', 'Norwegian': 'no',
    
    # 希伯来语
    'עברית': 'he', '希伯来语': 'he', 'he': 'he', 'Hebrew': 'he',
    
    # 罗马尼亚语
    'Română': 'ro', '罗马尼亚语': 'ro', 'ro': 'ro', 'Romanian': 'ro',
    
    # 保加利亚语
    'Български': 'bg', '保加利亚语': 'bg', 'bg': 'bg', 'Bulgarian': 'bg',
    
    # 乌克兰语
    'Українська': 'uk', '乌克兰语': 'uk', 'uk': 'uk', 'Ukrainian': 'uk',
    
    # 斯洛伐克语
    'Slovenčina': 'sk', '斯洛伐克语': 'sk', 'sk': 'sk', 'Slovak': 'sk',
    
    # 克罗地亚语
    'Hrvatski': 'hr', '克罗地亚语': 'hr', 'hr': 'hr', 'Croatian': 'hr',
    
    # 塞尔维亚语
    'Српски': 'sr', '塞尔维亚语': 'sr', 'sr': 'sr', 'Serbian': 'sr',
    
    # 斯洛文尼亚语
    'Slovenščina': 'sl', '斯洛文尼亚语': 'sl', 'sl': 'sl', 'Slovenian': 'sl',
    
    # 立陶宛语
    'Lietuvių': 'lt', '立陶宛语': 'lt', 'lt': 'lt', 'Lithuanian': 'lt',
    
    # 拉脱维亚语
    'Latviešu': 'lv', '拉脱维亚语': 'lv', 'lv': 'lv', 'Latvian': 'lv',
    
    # 爱沙尼亚语
    'Eesti': 'et', '爱沙尼亚语': 'et', 'et': 'et', 'Estonian': 'et',
    
    # 马来语
    'Bahasa Melayu': 'ms', '马来语': 'ms', 'ms': 'ms', 'Malay': 'ms',
    
    # 印尼语
    'Bahasa Indonesia': 'id', '印尼语': 'id', 'id': 'id', 'Indonesian': 'id',
    
    # 菲律宾语
    'Tagalog': 'tl', '菲律宾语': 'tl', 'tl': 'tl', 'Filipino': 'tl',
    
    # 波斯语
    'فارسی': 'fa', '波斯语': 'fa', 'fa': 'fa', 'Persian': 'fa',
    
    # 乌尔都语
    'اردو': 'ur', '乌尔都语': 'ur', 'ur': 'ur', 'Urdu': 'ur',
    
    # 孟加拉语
    'বাংলা': 'bn', '孟加拉语': 'bn', 'bn': 'bn', 'Bengali': 'bn',
}
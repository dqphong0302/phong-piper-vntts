"""
Enhanced Vietnamese Text Processor v2
Inspired by NghiTTS (github.com/nghimestudio/nghitts).
Handles: numbers, dates, time, currency, percentages, decimals, phone,
ordinals, Roman numerals, measurement units, ranges with units,
non-Vietnamese word transliteration, acronyms, text cleanup.
"""
import re
import os
import csv
import logging
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Number words ──────────────────────────────────────────────────────
DIGITS = {str(i): w for i, w in enumerate(
    ['không','một','hai','ba','bốn','năm','sáu','bảy','tám','chín']
)}

def number_to_words(num_str: str) -> str:
    num_str = num_str.lstrip('0') or '0'
    if num_str.startswith('-'):
        return 'âm ' + number_to_words(num_str[1:])
    try:
        n = int(num_str)
    except ValueError:
        return num_str
    if n == 0: return 'không'
    if n < 10: return DIGITS[str(n)]
    if n < 20:
        teens = {10:'mười',11:'mười một',12:'mười hai',13:'mười ba',14:'mười bốn',
                 15:'mười lăm',16:'mười sáu',17:'mười bảy',18:'mười tám',19:'mười chín'}
        return teens[n]
    if n < 100:
        t, u = divmod(n, 10)
        tens_w = DIGITS[str(t)] + ' mươi'
        if u == 0: return tens_w
        if u == 1: return tens_w + ' mốt'
        if u == 4: return tens_w + ' tư'
        if u == 5: return tens_w + ' lăm'
        return tens_w + ' ' + DIGITS[str(u)]
    if n < 1000:
        h, rem = divmod(n, 100)
        r = DIGITS[str(h)] + ' trăm'
        if rem == 0: return r
        if rem < 10: return r + ' lẻ ' + DIGITS[str(rem)]
        return r + ' ' + number_to_words(str(rem))

    units = [(1_000_000_000, 'tỷ'), (1_000_000, 'triệu'), (1_000, 'nghìn')]
    for div, name in units:
        if n >= div:
            q, rem = divmod(n, div)
            r = number_to_words(str(q)) + ' ' + name
            if rem == 0: return r
            if rem < 10: return r + ' không trăm lẻ ' + DIGITS[str(rem)]
            if rem < 100: return r + ' không trăm ' + number_to_words(str(rem))
            return r + ' ' + number_to_words(str(rem))

    return ' '.join(DIGITS.get(d, d) for d in num_str)


# ── Data files ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent

def _load_csv(filename: str) -> dict:
    path = DATA_DIR / filename
    result = {}
    if not path.exists():
        logger.warning(f"Data file not found: {path}")
        return result
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = list(row.values())[0].strip().lower()
            val = list(row.values())[1].strip()
            result[key] = val
    return result

NON_VIETNAMESE_WORDS = _load_csv('non-vietnamese-words.csv')
ACRONYMS = _load_csv('acronyms.csv')

# ── Measurement units ─────────────────────────────────────────────────
UNIT_MAP = {
    'm':'mét','cm':'xăng-ti-mét','mm':'mi-li-mét','km':'ki-lô-mét',
    'dm':'đề-xi-mét','inch':'in',
    'kg':'ki-lô-gam','g':'gam','mg':'mi-li-gam','tấn':'tấn',
    'ml':'mi-li-lít','l':'lít','lít':'lít',
    'm²':'mét vuông','m2':'mét vuông','km²':'ki-lô-mét vuông','km2':'ki-lô-mét vuông',
    'ha':'héc-ta','cm²':'xăng-ti-mét vuông','cm2':'xăng-ti-mét vuông',
    'm³':'mét khối','m3':'mét khối','cm³':'xăng-ti-mét khối','cm3':'xăng-ti-mét khối',
    'km/h':'ki-lô-mét trên giờ','m/s':'mét trên giây',
    '°C':'độ C','°F':'độ F',
    'kW':'ki-lô-oát','MW':'mê-ga-oát','W':'oát',
    'kWh':'ki-lô-oát giờ','MWh':'mê-ga-oát giờ',
    'V':'vôn','kV':'ki-lô-vôn','A':'am-pe','mA':'mi-li-am-pe',
    'Hz':'héc','kHz':'ki-lô-héc','MHz':'mê-ga-héc','GHz':'gi-ga-héc',
    'dB':'đề-xi-ben','cal':'ca-lo','kcal':'ki-lô ca-lo',
    'Mbps':'mê-ga-bít trên giây','Gbps':'gi-ga-bít trên giây',
    'MB':'mê-ga-bai','GB':'gi-ga-bai','TB':'tê-ra-bai','KB':'ki-lô-bai',
}

# ── Ordinal map ───────────────────────────────────────────────────────
ORDINAL_MAP = {
    '1':'nhất','2':'hai','3':'ba','4':'tư','5':'năm',
    '6':'sáu','7':'bảy','8':'tám','9':'chín','10':'mười'
}

# ── Roman Numerals ────────────────────────────────────────────────────
ROMAN_VALUES = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def _roman_to_arabic(s: str) -> int | None:
    if not s or not all(c in ROMAN_VALUES for c in s):
        return None
    total = 0
    prev = 0
    for c in reversed(s):
        v = ROMAN_VALUES[c]
        total += v if v >= prev else -v
        prev = v
    if total < 1 or total > 30:
        return None
    return total


# ═══════════════════════════════════════════════════════════════════════
# Processing pipeline
# ═══════════════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Remove emojis, normalize unicode, clean whitespace."""
    text = unicodedata.normalize('NFC', text)
    # Remove emojis
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF\U00002600-\U000026FF\U0000FE00-\U0000FE0F"
        "\U0000200D\U00002B50\U00002B55]+", flags=re.UNICODE)
    text = emoji_pattern.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _replace_acronyms(text: str) -> str:
    """Replace acronyms with transliterations."""
    for acr, trans in ACRONYMS.items():
        text = re.sub(r'\b' + re.escape(acr) + r'\b', trans, text, flags=re.IGNORECASE)
    return text

def _replace_non_vietnamese(text: str) -> str:
    """Replace non-Vietnamese words with phonetic transliterations."""
    for word, trans in NON_VIETNAMESE_WORDS.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', trans, text, flags=re.IGNORECASE)
    return text

def _remove_thousand_separators(text: str) -> str:
    """1.000.000 → 1000000 (Vietnamese dot as thousand separator)."""
    return re.sub(r'(\d{1,3}(?:\.\d{3})+)(?=\s|$|[^\d.,])', 
                  lambda m: m.group().replace('.', ''), text)

def _convert_ranges_with_units(text: str) -> str:
    """1-10kg → 1 đến 10 ki-lô-gam, 1/10kg → 1 phần 10 ki-lô-gam."""
    # Sort by length desc to match longer units first
    units_sorted = sorted(UNIT_MAP.keys(), key=len, reverse=True)
    units_pattern = '|'.join(re.escape(u) for u in units_sorted)
    
    # Ranges: 1-10kg
    text = re.sub(
        r'(\d+)\s*[-–—]\s*(\d+)\s*(' + units_pattern + r')(?![a-zà-ỹ])',
        lambda m: f"{m.group(1)} đến {m.group(2)} {UNIT_MAP.get(m.group(3), m.group(3))}",
        text, flags=re.IGNORECASE
    )
    # Fractions: 1/10kg
    text = re.sub(
        r'(\d+)/(\d+)\s*(' + units_pattern + r')(?![a-zà-ỹ])',
        lambda m: f"{m.group(1)} phần {m.group(2)} {UNIT_MAP.get(m.group(3), m.group(3))}",
        text, flags=re.IGNORECASE
    )
    return text

def _convert_percentage(text: str) -> str:
    """Convert percentages: 3-5% → ba đến năm phần trăm, 3,2% → ba phẩy hai phần trăm."""
    # Percentage ranges
    text = re.sub(r'(\d+)\s*[-–—]\s*(\d+)\s*%',
                  lambda m: f"{number_to_words(m.group(1))} đến {number_to_words(m.group(2))} phần trăm", text)
    # Decimal percentages
    text = re.sub(r'(\d+),(\d+)\s*%',
                  lambda m: f"{number_to_words(m.group(1))} phẩy {number_to_words(m.group(2))} phần trăm", text)
    # Simple percentages
    text = re.sub(r'(\d+)\s*%', lambda m: number_to_words(m.group(1)) + ' phần trăm', text)
    return text

def _convert_currency(text: str) -> str:
    """Currency: 50000đ → năm mươi nghìn đồng, $100 → một trăm đô la."""
    def vnd(m):
        return number_to_words(m.group(1).replace(',','')) + ' đồng'
    def usd(m):
        return number_to_words(m.group(1).replace(',','')) + ' đô la'
    
    text = re.sub(r'(\d+(?:,\d+)?)\s*(?:đồng|VND|vnđ)\b', vnd, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:,\d+)?)đ(?![a-zà-ỹ])', vnd, text, flags=re.IGNORECASE)
    text = re.sub(r'\$\s*(\d+(?:,\d+)?)', usd, text)
    text = re.sub(r'(\d+(?:,\d+)?)\s*(?:USD|\$)', usd, text, flags=re.IGNORECASE)
    return text

def _convert_date(text: str) -> str:
    """Dates: 25/12/2024 → ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi tư."""
    def valid_date(d, m, y=None):
        dd, mm = int(d), int(m)
        ok = 1 <= dd <= 31 and 1 <= mm <= 12
        if y: ok = ok and 1000 <= int(y) <= 9999
        return ok
    
    # Date ranges: ngày dd-dd/mm/yyyy
    text = re.sub(r'ngày\s+(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*[/-]\s*(\d{1,2})(?:\s*[/-]\s*(\d{4}))?',
                  lambda m: f"ngày {number_to_words(m.group(1))} đến {number_to_words(m.group(2))} tháng {number_to_words(m.group(3))}" +
                            (f" năm {number_to_words(m.group(4))}" if m.group(4) else '') if valid_date(m.group(1), m.group(3), m.group(4)) else m.group(),
                  text)
    # Full date dd/mm/yyyy
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
                  lambda m: f"ngày {number_to_words(m.group(1))} tháng {number_to_words(m.group(2))} năm {number_to_words(m.group(3))}" if valid_date(m.group(1), m.group(2), m.group(3)) else m.group(),
                  text)
    # dd/mm
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})(?!\d)',
                  lambda m: f"ngày {number_to_words(m.group(1))} tháng {number_to_words(m.group(2))}" if valid_date(m.group(1), m.group(2)) else m.group(),
                  text)
    return text

def _convert_year_range(text: str) -> str:
    """Year ranges: 1873-1907 → ... đến ..."""
    return re.sub(r'(\d{4})\s*[-–—]\s*(\d{4})',
                  lambda m: f"{number_to_words(m.group(1))} đến {number_to_words(m.group(2))}", text)

def _convert_time(text: str) -> str:
    """Time: 14:30 → mười bốn giờ ba mươi phút."""
    def time_repl(m):
        h, mn = m.group(1), m.group(2)
        s = m.group(3) if m.lastindex >= 3 and m.group(3) else None
        r = number_to_words(h) + ' giờ ' + number_to_words(mn) + ' phút'
        if s: r += ' ' + number_to_words(s) + ' giây'
        return r
    text = re.sub(r'(\d{1,2}):(\d{2}):(\d{2})', time_repl, text)
    text = re.sub(r'(\d{1,2}):(\d{2})(?!:)', time_repl, text)
    return text

def _convert_ordinal(text: str) -> str:
    """Ordinals: thứ 2 → thứ hai, chương 3 → chương ba."""
    def repl(m):
        prefix, num = m.group(1), m.group(2)
        return prefix + ' ' + (ORDINAL_MAP.get(num) or number_to_words(num))
    return re.sub(r'(thứ|lần|bước|phần|chương|tập|số)\s*(\d+)', repl, text, flags=re.IGNORECASE)

def _convert_roman(text: str) -> str:
    """Roman numerals I-XXX → Arabic."""
    def repl(m):
        roman = m.group()
        if roman != roman.upper():
            return roman
        val = _roman_to_arabic(roman)
        return str(val) if val else roman
    return re.sub(r'\b([IVXLCDM]{1,6})\b', repl, text)

def _convert_phone(text: str) -> str:
    """Phone: 0912345678 → không chín một hai ba bốn năm sáu bảy tám."""
    def digit_by_digit(m):
        return ' '.join(DIGITS.get(d, d) for d in m.group() if d.isdigit())
    text = re.sub(r'0\d{9,10}', digit_by_digit, text)
    text = re.sub(r'\+84\d{9,10}', digit_by_digit, text)
    return text

def _convert_decimal(text: str) -> str:
    """Decimal: 7,27 → bảy phẩy hai mươi bảy."""
    return re.sub(r'(\d+),(\d+)(?=\s|$|[^\d,])',
                  lambda m: f"{number_to_words(m.group(1))} phẩy {number_to_words(m.group(2).lstrip('0') or '0')}",
                  text)

def _convert_measurement_units(text: str) -> str:
    """Replace measurement units after numbers."""
    units_sorted = sorted(UNIT_MAP.keys(), key=len, reverse=True)
    units_pattern = '|'.join(re.escape(u) for u in units_sorted)
    return re.sub(
        r'(\d+)\s*(' + units_pattern + r')(?![a-zà-ỹ])',
        lambda m: f"{m.group(1)} {UNIT_MAP.get(m.group(2), m.group(2))}",
        text
    )

def _convert_standalone_numbers(text: str) -> str:
    """Convert remaining standalone numbers."""
    return re.sub(r'\b\d+\b', lambda m: number_to_words(m.group()), text)


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════

def process_vietnamese_text(text: str) -> str:
    """Full Vietnamese text preprocessing pipeline."""
    if not text or not text.strip():
        return text
    
    original = text
    
    # 1. Clean text
    text = _clean_text(text)
    
    # 2. Replace acronyms & non-Vietnamese words
    text = _replace_acronyms(text)
    text = _replace_non_vietnamese(text)
    
    # 3. Normalize thousand separators (dots)
    text = _remove_thousand_separators(text)
    
    # 4. Convert ranges with units BEFORE individual conversions
    text = _convert_ranges_with_units(text)
    
    # 5. Percentages, currency, dates, time (order matters!)
    text = _convert_percentage(text)
    text = _convert_currency(text)
    text = _convert_date(text)
    text = _convert_year_range(text)
    text = _convert_time(text)
    
    # 6. Ordinals, Roman, phone, decimal
    text = _convert_ordinal(text)
    text = _convert_roman(text)
    text = _convert_phone(text)
    text = _convert_decimal(text)
    
    # 7. Measurement units, then standalone numbers
    text = _convert_measurement_units(text)
    text = _convert_standalone_numbers(text)
    
    # 8. Final cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    if text != original:
        logger.info(f"[VN Preprocessor] \"{original}\" → \"{text}\"")
    
    return text

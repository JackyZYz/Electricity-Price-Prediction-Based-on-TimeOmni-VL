import re


def extract_between_tags(text: str, tag: str = "think") -> str:
    pattern = re.compile(rf"\u003c{tag}\u003e(.*?)\u003c/{tag}\u003e", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def format_price(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"

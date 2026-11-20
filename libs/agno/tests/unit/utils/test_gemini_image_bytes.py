from agno.media import Image
from agno.utils.gemini import format_image_for_message

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def test_format_image_for_message_bytes_content_returns_raw_bytes():
    result = format_image_for_message(Image(content=PNG))
    assert result is not None
    assert isinstance(result["data"], bytes), f"expected raw bytes, got {type(result['data']).__name__}"
    assert result["data"] == PNG


def test_format_image_for_message_url_returns_raw_bytes(monkeypatch):
    image = Image(url="https://example.com/a.png")
    monkeypatch.setattr(Image, "get_content_bytes", lambda self: PNG)
    result = format_image_for_message(image)
    assert result is not None
    assert isinstance(result["data"], bytes), f"expected raw bytes, got {type(result['data']).__name__}"
    assert result["data"] == PNG

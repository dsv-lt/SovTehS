"""Общие функции предобработки текста.

Модуль используется и в исследовательских ноутбуках, и в продакшен-сервисе,
что гарантирует идентичную обработку текста при обучении и инференсе.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

# Основное регулярное выражение из ТЗ («Напиши текс в стиле автора Платонов С»).
# Запрет перевода строк исключает многострочные инъекции команд через имя автора.
AUTHOR_RE_PRIMARY = re.compile(r"стиле автора ([^\"\r\n]+)")
# В датасете встречаются и другие шаблоны запроса:
#   «Придумай отрывок из книги Анна Каренина автора Толстой Л»,
#   «Напиши текс похожий на текст автора Чехов А».
# Во всех шаблонах имя автора стоит в конце после слова «автора», поэтому
# резервное выражение позволяет сохранить 100 % записей и все классы.
AUTHOR_RE_FALLBACK = re.compile(r"автора\s+([^\"\r\n]+?)\s*$")

_LITERAL_ESCAPES = (
    ("\\r\\n", "\n"),
    ("\\n", "\n"),
    ("\\r", "\n"),
    ("\\t", " "),
    ('\\"', '"'),
    ("\\'", "'"),
)
_SPACES_RE = re.compile(r"[ \t\u00a0\u2009]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_WS_RE = re.compile(r"\s+")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]")


def extract_author(user_message: str) -> str | None:
    """Извлекает имя автора (метку класса) из сообщения пользователя."""
    match = AUTHOR_RE_PRIMARY.search(user_message) or AUTHOR_RE_FALLBACK.search(user_message)
    return match.group(1).strip() if match else None


def clean_text(text: str) -> str:
    """Очистка литературного фрагмента.

    * исправление «битых» (буквальных) экранирований ``\\n``, ``\\"`` и т.п.;
    * нормализация переносов строк ``\\r\\n`` -> ``\\n``;
    * удаление управляющих символов и дублирующихся пробелов;
    * удаление пустых строк.
    """
    if not text:
        return ""
    for src, dst in _LITERAL_ESCAPES:
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    lines = (line.strip() for line in text.split("\n"))
    text = "\n".join(line for line in lines if line)
    return _MULTI_NL_RE.sub("\n", text).strip()


def normalize_for_model(text: str) -> str:
    """Приводит текст к виду, подаваемому в классификатор.

    Переносы строк в датасете — артефакт вёрстки исходных файлов, а не стиль
    автора, а запрос пользователя обычно приходит одной строкой. Поэтому все
    пробельные символы схлопываются в один пробел.
    """
    return _WS_RE.sub(" ", clean_text(text)).strip()


def chunk_text(text: str, chunk_chars: int = 1000, min_chars: int = 200) -> list[str]:
    """Нарезает длинный текст на фрагменты ~``chunk_chars`` символов.

    Разбиение идёт по границам слов; хвостовой фрагмент короче ``min_chars``
    присоединяется к предыдущему (или отбрасывается, если он единственный и
    слишком короткий — кроме случая, когда весь текст короткий).
    Монолитные токены без пробелов нарезаются принудительно для защиты от DoS.
    """
    normalized = normalize_for_model(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        if not word:
            continue
        # Защита от сверхдлинных токенов без пробелов
        while len(word) > chunk_chars:
            head, word = word[:chunk_chars], word[chunk_chars:]
            chunks.append(head)

        current.append(word)
        size += len(word) + 1
        if size >= chunk_chars:
            chunks.append(" ".join(current))
            current, size = [], 0
    if current:
        tail = " ".join(current)
        if len(tail) >= min_chars or not chunks:
            chunks.append(tail)
        else:
            chunks[-1] = f"{chunks[-1]} {tail}"
    return chunks


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    """Построчное чтение JSON Lines файла датасета."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_record(record: dict) -> tuple[str | None, str]:
    """Возвращает (автор, текст) из записи с диалоговой структурой messages."""
    author, text = None, ""
    for message in record.get("messages", []):
        if message.get("role") == "user" and author is None:
            author = extract_author(message.get("content", ""))
        elif message.get("role") == "assistant" and not text:
            text = message.get("content", "")
    return author, text

"""上传文件文本提取：按扩展名解析纯文本 / Word(docx) / PDF。

背景：社区投稿接口原先把文件字节直接 `raw.decode("utf-8", errors="ignore")`，
导致两类乱码：
- GBK/GB2312 编码的中文文本（Windows 记事本保存常见）被按 UTF-8 解码，中文全变乱码；
- .docx/.pdf 等二进制格式被当作 UTF-8 文本读出 "PK\x03\x04..." 之类的垃圾。

本模块按文件类型分派提取方式，并对解码结果做"疑似二进制"校验；
任何提取失败都抛出 FileParseError（message 可直接展示给用户），
保证不会再把乱码写入知识库。
"""

from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO

# 与前端 accept 一致（.doc 旧格式不支持解析，会给出明确提示）
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".json", ".csv", ".log", ".text"}
DOCX_EXTENSION = ".docx"
PDF_EXTENSION = ".pdf"
LEGACY_DOC_EXTENSION = ".doc"
SUPPORTED_HINT = ".txt/.md/.json/.csv/.docx/.pdf"

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_UTF8_BOM = b"\xef\xbb\xbf"
_UTF16_LE_BOM = b"\xff\xfe"
_UTF16_BE_BOM = b"\xfe\xff"

# 正常文本中不应出现的控制字符（保留 \t \n \r），命中即视为二进制内容
_BINARY_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class FileParseError(ValueError):
    """文件无法提取出可用文本时抛出，message 为可直接展示给用户的提示。"""


def _guard_decoded_text(text: str, encoding: str) -> str:
    """对解码结果做疑似二进制校验，避免把二进制数据当文本入库。"""
    if _BINARY_CONTROL_RE.search(text):
        raise FileParseError(f"文件内容疑似二进制数据，无法按 {encoding} 文本读取。")
    return text


def decode_text_bytes(raw: bytes) -> str:
    """把纯文本文件字节解码为 str。

    解码顺序：
    1. UTF-16（仅当存在 BOM，避免无 BOM 的普通文本被误判）；
    2. UTF-8（含 BOM，utf-8-sig 会顺带去掉 BOM）；
    3. GB18030（兼容 GBK/GB2312，Windows 中文文本常见编码）。
    全部失败时抛 FileParseError。
    """
    if not raw:
        return ""

    if raw.startswith(_UTF16_LE_BOM) or raw.startswith(_UTF16_BE_BOM):
        return _guard_decoded_text(_decode(raw, "utf-16"), "UTF-16")

    if raw.startswith(_UTF8_BOM):
        return _guard_decoded_text(_decode(raw, "utf-8-sig"), "UTF-8")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return _guard_decoded_text(text, "UTF-8")

    try:
        text = raw.decode("gb18030")
    except UnicodeDecodeError:
        raise FileParseError(
            "无法识别的文件编码，请将文件另存为 UTF-8 编码（.txt）后再上传。"
        ) from None
    return _guard_decoded_text(text, "GB18030")


def _decode(raw: bytes, encoding: str) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as e:
        raise FileParseError(f"文件 {encoding} 解码失败，文件可能已损坏。") from e


def extract_docx_text(raw: bytes) -> str:
    """从 .docx（ZIP + WordprocessingML）中提取段落文本，纯标准库实现。"""
    try:
        with zipfile.ZipFile(BytesIO(raw)) as zf:
            xml_name = None
            for name in zf.namelist():
                if name.startswith("word/") and name.endswith(".xml"):
                    base = name.rsplit("/", 1)[-1]
                    if base == "document.xml" or base.startswith("document"):
                        xml_name = name
                        break
            if xml_name is None:
                raise FileParseError("不是有效的 Word 文档（缺少 document.xml）。")
            doc_xml = zf.read(xml_name)
    except FileParseError:
        raise
    except (zipfile.BadZipFile, OSError, KeyError, RuntimeError):
        raise FileParseError("无法解析该文件：不是有效的 .docx（Word）文档。") from None

    try:
        root = ET.fromstring(doc_xml)
    except ET.ParseError:
        raise FileParseError("Word 文档内容解析失败，文件可能已损坏。") from None

    paragraphs: list[str] = []
    for para in root.iter(_W_NS + "p"):
        runs = [t.text or "" for t in para.iter(_W_NS + "t")]
        line = "".join(runs).strip()
        if line:
            paragraphs.append(line)

    text = "\n".join(paragraphs).strip()
    if not text:
        raise FileParseError("该 Word 文档中没有可提取的文字内容。")
    return text


def extract_pdf_text(raw: bytes) -> str:
    """从 .pdf 中提取文本（pypdf，支持 ToUnicode 映射，中文 PDF 不乱码）。"""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        raise FileParseError("服务器缺少 PDF 解析组件，暂时无法处理 .pdf 文件。") from None

    try:
        reader = PdfReader(BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        raise FileParseError(
            "无法解析该 PDF 文件（可能已加密或损坏）。"
        ) from None

    text = text.strip()
    if not text:
        raise FileParseError(
            "该 PDF 没有可提取的文字内容（扫描/图片型 PDF 暂不支持 OCR）。"
        )
    return text


def extract_text_from_file(filename: str | None, raw: bytes) -> str:
    """按扩展名从上传文件提取文本；失败时抛 FileParseError。

    - .txt/.md/.json/.csv 等：按编码探测解码；
    - .docx：标准库解析 ZIP 内 document.xml；
    - .pdf：pypdf 提取；
    - .doc 及未知二进制类型：给出明确提示，而不是产出乱码。
    """
    if not raw:
        return ""

    ext = os.path.splitext(filename or "")[1].lower()

    if ext in TEXT_EXTENSIONS or ext == "":
        return decode_text_bytes(raw)
    if ext == DOCX_EXTENSION:
        return extract_docx_text(raw)
    if ext == PDF_EXTENSION:
        return extract_pdf_text(raw)
    if ext == LEGACY_DOC_EXTENSION:
        raise FileParseError(
            "暂不支持旧版 Word 文档（.doc）。请先用 Word/WPS 另存为 .docx 或 .txt 后再上传。"
        )

    raise FileParseError(
        f"暂不支持“{ext or '未知类型'}”文件。请上传文本文件：{SUPPORTED_HINT}。"
    )
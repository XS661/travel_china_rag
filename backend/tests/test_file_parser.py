"""文件解析与编码探测测试：GBK/BOM/docx/pdf/二进制垃圾 等上传场景。

运行方式（仓库根目录）：
    uv run python -m unittest discover -s backend/tests -v
"""

import io
import unittest
import zipfile

from backend.file_parser import (
    FileParseError,
    decode_text_bytes,
    extract_docx_text,
    extract_pdf_text,
    extract_text_from_file,
)


def build_docx(body_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body_xml}</w:body></w:document>",
        )
    return buf.getvalue()


def build_pdf(text: str) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    content = DecodedStreamObject()
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content.set_data(f"BT /F1 24 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(content)
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    page[NameObject("/Resources")] = writer._add_object(
        DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    )
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class DecodeTextTests(unittest.TestCase):
    def test_utf8_plain(self):
        self.assertEqual(decode_text_bytes("成都宽窄巷子".encode("utf-8")), "成都宽窄巷子")

    def test_utf8_with_bom(self):
        raw = b"\xef\xbb\xbf" + "成都宽窄巷子".encode("utf-8")
        self.assertEqual(decode_text_bytes(raw), "成都宽窄巷子")

    def test_utf16_with_bom(self):
        self.assertEqual(
            decode_text_bytes("重庆洪崖洞夜景很美".encode("utf-16")), "重庆洪崖洞夜景很美"
        )

    def test_gbk_text(self):
        # Windows 记事本“ANSI”保存即 GBK/GB2312，曾经会被 UTF-8 解码成乱码
        raw = "我在春熙路吃火锅很好吃，晚上可以看夜景。".encode("gbk")
        self.assertEqual(
            decode_text_bytes(raw), "我在春熙路吃火锅很好吃，晚上可以看夜景。"
        )

    def test_gb18030_text(self):
        raw = "打卡张家界玻璃栈道，风景震撼。".encode("gb18030")
        self.assertEqual(decode_text_bytes(raw), "打卡张家界玻璃栈道，风景震撼。")

    def test_empty_bytes(self):
        self.assertEqual(decode_text_bytes(b""), "")

    def test_binary_garbage_is_rejected(self):
        with self.assertRaises(FileParseError):
            decode_text_bytes(bytes(range(256)) * 4)
        with self.assertRaises(FileParseError):
            decode_text_bytes(b"\x00\x01\x02binary\x00\x03")


class DocxExtractTests(unittest.TestCase):
    def test_extracts_chinese_paragraphs(self):
        body = (
            "<w:p><w:r><w:t>成都宽窄巷子值得一逛</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>春熙路晚上很热闹</w:t></w:r></w:p>"
        )
        text = extract_docx_text(build_docx(body))
        self.assertEqual(text, "成都宽窄巷子值得一逛\n春熙路晚上很热闹")

    def test_not_a_zip_is_rejected(self):
        with self.assertRaises(FileParseError):
            extract_docx_text(b"PK\x03\x04 not really a zip file")

    def test_zip_without_document_xml_is_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("word/document2.xml", "<w:document/>")
        with self.assertRaises(FileParseError):
            extract_docx_text(buf.getvalue())

    def test_docx_without_text_is_rejected(self):
        with self.assertRaises(FileParseError):
            extract_docx_text(build_docx("<w:p><w:r><w:t>  </w:t></w:r></w:p>"))


class PdfExtractTests(unittest.TestCase):
    def test_extracts_plain_text_pdf(self):
        pdf = build_pdf("Hello Chengdu, Dujiangyan is great")
        self.assertEqual(extract_pdf_text(pdf), "Hello Chengdu, Dujiangyan is great")

    def test_garbage_pdf_is_rejected(self):
        with self.assertRaises(FileParseError):
            extract_pdf_text(b"%PDF-1.4 not a real pdf at all")


class DispatchTests(unittest.TestCase):
    def test_txt_via_dispatch(self):
        self.assertEqual(
            extract_text_from_file("notes.txt", "成都旅游".encode("gbk")), "成都旅游"
        )

    def test_docx_via_dispatch(self):
        body = "<w:p><w:r><w:t>成都美食攻略</w:t></w:r></w:p>"
        self.assertEqual(extract_text_from_file("a.docx", build_docx(body)), "成都美食攻略")

    def test_pdf_via_dispatch(self):
        self.assertEqual(
            extract_text_from_file("a.pdf", build_pdf("Chengdu trip notes")),
            "Chengdu trip notes",
        )

    def test_legacy_doc_is_rejected_with_guidance(self):
        with self.assertRaises(FileParseError) as ctx:
            extract_text_from_file("old.doc", b"\xd0\xcf\xd4\xc4\xcf")
        self.assertIn("另存为", str(ctx.exception))

    def test_unknown_type_is_rejected(self):
        with self.assertRaises(FileParseError) as ctx:
            extract_text_from_file("photo.png", b"\x89PNG\r\n\x1a\n")
        self.assertIn("暂不支持", str(ctx.exception))

    def test_extensionless_plain_text_still_works(self):
        self.assertEqual(
            extract_text_from_file("README", "深圳莲花山公园".encode("gb18030")),
            "深圳莲花山公园",
        )


if __name__ == "__main__":
    unittest.main()
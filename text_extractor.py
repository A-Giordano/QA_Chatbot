import docx
import pdfplumber


class TextExtractor:
    _file_name: str = None
    _extraction_strategies = {
        "doc": "_extract_doc",
        "docx": "_extract_doc",
        "pdf": "_extract_pdf",
    }

    def __init__(self, file_name) -> None:
        self._file_name = file_name
        format = self._get_format(file_name)
        extraction_method = self._extraction_strategies[format]
        self.extract = getattr(self, extraction_method)
        pass

    def _extract_pdf(self) -> str:
        pdf = pdfplumber.open(self._file_name)
        text = "/n".join([page.extract_text() for page in pdf.pages])
        pdf.close()
        return text

    def _extract_doc(self) -> str:
        doc = docx.Document(self._file_name)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return "\n".join(fullText)

    def _get_format(self, file_name) -> str:
        format = (file_name.split("."))[-1]
        if format is None or format not in [
            format for format, method in self._extraction_strategies.items()
        ]:
            raise Exception(f"File format {format} is None or unsupported")
        return format

    def extract(self) -> str:
        # Valorized in constructor
        ...
import os
import sys
# Ensure parent directory is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from PIIshield import PIIPipeline

class PDFRedactionHandler:
    def __init__(self):
        self.pipeline = PIIPipeline()

    def process(self, input_path: str):
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return None
        _, ext = os.path.splitext(input_path.lower())
        output_path = self._get_output_path(input_path, ext)
        print(f"Processing: {input_path}")
        results = self.pipeline.process_document(input_path)
        if ext == '.pdf':
            # Check if we have a fully redacted PDF (text + faces)
            if 'final_pdf_path' in results:
                print(f"\n✓ Fully redacted PDF (text + faces): {results['final_pdf_path']}")
                return results['final_pdf_path']
            elif 'pdf_redacted_path' in results:
                print(f"\n✓ Text-redacted PDF: {results['pdf_redacted_path']}")
                print("ℹ Note: No faces detected or face redaction not applied")
                return results['pdf_redacted_path']
            else:
                print("✗ PDF redaction failed.")
                return None
        else:
            # Non-standardized (image) input
            if 'final_redacted_path' in results:
                print(f"Redacted image saved at: {results['final_redacted_path']}")
                return results['final_redacted_path']
            else:
                print("Image redaction failed.")
                return None

    def _get_output_path(self, input_path, ext):
        base = os.path.splitext(os.path.basename(input_path))[0]
        if ext == '.pdf':
            return os.path.join(os.path.dirname(input_path), f"redacted_{base}.pdf")
        else:
            return os.path.join(os.path.dirname(input_path), f"redacted_{base}.jpg")

if __name__ == "__main__":
    input_path = input("Enter the path to the file (PDF or image): ").strip()
    handler = PDFRedactionHandler()
    output_path = handler.process(input_path)
    if output_path:
        print(f"Output file: {output_path}")
    else:
        print("Redaction failed.")

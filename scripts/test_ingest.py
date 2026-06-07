import traceback
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.rag_app.api import ingest
    from src.rag_app.schema import IngestRequest, SourceDocument, Metadata

    req = IngestRequest(documents=[
        SourceDocument(id='doc1', text='This is a test document.', metadata=Metadata(source='local', title='Doc1'))
    ])

    print('Calling ingest...')
    result = ingest(req)
    print('Result:', result)
except Exception:
    traceback.print_exc()

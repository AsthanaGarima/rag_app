import traceback
import sys
from pathlib import Path

# ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.rag_app.api import query
    from src.rag_app.schema import QueryRequest

    req = QueryRequest(query='What is in the documents?')
    print('Calling query...')
    result = query(req)
    print('Result:', result)
except Exception:
    traceback.print_exc()

# Being in the repo root makes pytest put it on sys.path, so tests can `import stream`.
import os

os.environ.setdefault("TRACE", "0")   # tests must not litter logs/ with trace files

"""Small JSON, JSONL, Parquet and YAML helpers."""

import hashlib
import json
import os
from pathlib import Path


def read_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError("expected a list of records")
    return rows


def read_config(path):
    import yaml

    result = yaml.safe_load(Path(path).read_text())
    if not isinstance(result, dict):
        raise ValueError("expected a configuration mapping")
    return result


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def write_json(path, value):
    write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        pq.write_table(pa.Table.from_pylist(rows), tmp)
        tmp.replace(path)
    elif path.suffix == ".jsonl":
        write_text(
            path,
            "".join(
                json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows
            ),
        )
    else:
        write_json(path, rows)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class Cache:
    """Append completed requests; resume by input/model/settings content. One writer per file."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        if self.path.exists():
            for row in read_rows(self.path):
                self.records[row["key"]] = row["value"]

    def get_or_run(self, request, function):
        key = digest(request)
        if key not in self.records:
            result = function()
            line = json.dumps({"key": key, "value": result}, allow_nan=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(line + "\n")
                handle.flush()
            self.records[key] = result
        return self.records[key]

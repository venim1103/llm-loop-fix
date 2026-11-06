"""Exact sparse artifact replay, separate from automatic repair selection."""

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from safetensors import safe_open


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _layout(path):
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("Missing Safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= 1024 * 1024 or header_size + 8 > size:
            raise ValueError("Invalid or oversized Safetensors header")
        header = json.loads(
            source.read(header_size),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    with safe_open(str(path), framework="np") as reader:
        names = set(reader.keys())
        if names != set(header) - {"__metadata__"}:
            raise ValueError("Safetensors tensor-name mismatch")
        for name in names:
            view = reader.get_slice(name)
            if view.get_shape() != header[name]["shape"] or view.get_dtype() != header[name]["dtype"]:
                raise ValueError(f"Safetensors layout mismatch: {name}")
    return header, 8 + header_size


def _hash_range(source, offset, length):
    source.seek(offset)
    digest = hashlib.sha256()
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("Truncated tensor data")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def audit_safetensors(path: Path | str) -> dict:
    path = Path(path)
    header, data_start = _layout(path)
    report = {"size_bytes": path.stat().st_size, "tensors": {}}
    with path.open("rb") as source:
        report["file_sha256"] = _hash_range(source, 0, report["size_bytes"])
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            report["tensors"][name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "sha256": _hash_range(source, data_start + begin, end - begin),
            }
    return report


def _checked_bytes(path, expected_sha256, limit):
    if path.stat().st_size > limit:
        raise ValueError(f"Input exceeds byte cap: {path.name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Input hash mismatch: {path.name}")
    return data


def _sparse_arrays(data, tensor_count):
    expected = {
        f"t{index}_{suffix}.npy"
        for index in range(tensor_count)
        for suffix in ("idx", "val")
    }
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
            raise ValueError("Unexpected or duplicate NPZ entries")
        if sum(entry.file_size for entry in entries) > 16 * 1024 * 1024:
            raise ValueError("NPZ expansion exceeds byte cap")
        for entry in entries:
            with archive.open(entry) as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(member, max_header_size=4096)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(member, max_header_size=4096)
                else:
                    raise ValueError("Unsupported NPY header version")
                if len(shape) != 1 or dtype.hasobject or dtype.kind not in "iu":
                    raise ValueError("NPZ arrays must be one-dimensional numeric integers")
                if shape[0] * dtype.itemsize != entry.file_size - member.tell():
                    raise ValueError("NPZ array shape/byte mismatch")
                if entry.filename.endswith("_val.npy") and dtype != np.dtype("<u2"):
                    raise ValueError("Replacement values must be little-endian BF16 uint16 bits")
    with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=4096) as archive:
        return [
            (archive[f"t{index}_idx"], archive[f"t{index}_val"])
            for index in range(tensor_count)
        ]


def replay_sparse_edit(
    base_path: Path | str,
    edit_path: Path | str,
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    expected_base_sha256: str,
    expected_edit_sha256: str,
    expected_manifest_sha256: str,
) -> dict:
    base_path, edit_path, manifest_path, output_path = map(
        Path, (base_path, edit_path, manifest_path, output_path)
    )
    if output_path.exists() or output_path.is_symlink() or output_path.resolve() == base_path.resolve():
        raise FileExistsError("Replay output must be a fresh path, separate from the baseline")
    manifest = json.loads(
        _checked_bytes(manifest_path, expected_manifest_sha256, 1024 * 1024),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not 1 <= len(tensors) <= 2:
        raise ValueError("Replay requires one or two sparse tensors")
    if type(manifest.get("changed_tensors")) is not int or manifest["changed_tensors"] != len(tensors):
        raise ValueError("Manifest tensor count mismatch")
    if type(manifest.get("changed_entries_total")) is not int:
        raise ValueError("Invalid manifest changed-entry total")
    names = [tensor["name"] for tensor in tensors]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate manifest target")
    for tensor in tensors:
        shape = tensor.get("shape")
        if (
            tensor.get("dtype") != "bfloat16"
            or tensor.get("form") != "sparse"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
        ):
            raise ValueError("Unsupported sparse tensor dtype/form/shape")
        for field in ("changed_entries", "changed_rows", "changed_columns"):
            if type(tensor.get(field)) is not int or tensor[field] < 0:
                raise ValueError(f"Invalid manifest field: {field}")
    arrays = _sparse_arrays(
        _checked_bytes(edit_path, expected_edit_sha256, 2 * 1024 * 1024), len(tensors)
    )
    baseline = audit_safetensors(base_path)
    if baseline["file_sha256"] != expected_base_sha256:
        raise ValueError("Baseline file hash mismatch")
    header, data_start = _layout(base_path)
    for tensor in tensors:
        original = baseline["tensors"].get(tensor["name"])
        if original != {
            "dtype": "BF16", "shape": tensor["shape"], "sha256": tensor["base_sha256"]
        }:
            raise ValueError(f"Baseline tensor mismatch: {tensor['name']}")

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".staging", dir=output_path.parent
    )
    os.close(descriptor)
    staging = Path(temporary)
    measurements = []
    try:
        shutil.copyfile(base_path, staging)
        with staging.open("r+b") as destination:
            for tensor, (indices, values) in zip(tensors, arrays):
                begin, end = header[tensor["name"]]["data_offsets"]
                destination.seek(data_start + begin)
                original_bits = np.frombuffer(destination.read(end - begin), dtype="<u2")
                if (
                    len(indices) != len(values)
                    or not len(indices)
                    or np.any(indices < 0)
                    or np.any(indices >= original_bits.size)
                    or len(np.unique(indices)) != len(indices)
                ):
                    raise ValueError("Invalid, duplicate, or out-of-bounds sparse indices")
                if np.any((values & 0x7F80) == 0x7F80):
                    raise ValueError("Nonfinite BF16 replacement")
                edited_bits = original_bits.copy()
                edited_bits[indices.astype(np.intp)] = values
                difference = (edited_bits != original_bits).reshape(tensor["shape"])
                measured = {
                    "name": tensor["name"],
                    "changed_entries": int(np.count_nonzero(difference)),
                    "changed_rows": int(np.count_nonzero(np.any(difference, axis=1))),
                    "changed_columns": int(np.count_nonzero(np.any(difference, axis=0))),
                }
                if any(measured[field] != tensor[field] for field in measured if field != "name"):
                    raise ValueError("Stored-bit support/count mismatch")
                data = edited_bits.tobytes()
                edited_sha256 = hashlib.sha256(data).hexdigest()
                if edited_sha256 != tensor["edited_sha256"]:
                    raise ValueError("Reconstructed tensor hash mismatch")
                measured["base_sha256"] = hashlib.sha256(original_bits.tobytes()).hexdigest()
                measured["edited_sha256"] = edited_sha256
                destination.seek(data_start + begin)
                destination.write(data)
                measurements.append(measured)
            destination.flush()
            os.fsync(destination.fileno())
        if sum(item["changed_entries"] for item in measurements) != manifest["changed_entries_total"]:
            raise ValueError("Measured changed-entry total mismatch")
        fresh = audit_safetensors(staging)
        if _layout(staging) != (header, data_start) or set(fresh["tensors"]) != set(baseline["tensors"]):
            raise ValueError("Fresh-load header/tensor-name mismatch")
        expected_tensors = {name: dict(info) for name, info in baseline["tensors"].items()}
        for tensor in tensors:
            expected_tensors[tensor["name"]]["sha256"] = tensor["edited_sha256"]
        if fresh["tensors"] != expected_tensors:
            raise ValueError("Fresh-load tensor preservation/hash mismatch")
        if base_path.stat().st_size != baseline["size_bytes"]:
            raise ValueError("Immutable baseline size changed during replay")
        with base_path.open("rb") as source:
            if _hash_range(source, 0, baseline["size_bytes"]) != expected_base_sha256:
                raise ValueError("Immutable baseline changed during replay")
        os.link(staging, output_path)
        return {
            "claim_scope": "stored_tensor_bytes_only",
            "base_file_sha256": expected_base_sha256,
            "edit_file_sha256": expected_edit_sha256,
            "manifest_file_sha256": expected_manifest_sha256,
            "output_file_sha256": fresh["file_sha256"],
            "tensor_count": len(fresh["tensors"]),
            "unchanged_tensors": len(fresh["tensors"]) - len(tensors),
            "changed_entries_total": sum(item["changed_entries"] for item in measurements),
            "tensors": measurements,
        }
    finally:
        staging.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Replay a pinned sparse BF16 artifact without inference")
    for name in ("base", "edit", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("base", "edit", "manifest"):
        parser.add_argument(f"--{name}-sha256", required=True)
    arguments = parser.parse_args()
    report = replay_sparse_edit(
        arguments.base, arguments.edit, arguments.manifest, arguments.output,
        expected_base_sha256=arguments.base_sha256,
        expected_edit_sha256=arguments.edit_sha256,
        expected_manifest_sha256=arguments.manifest_sha256,
    )
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

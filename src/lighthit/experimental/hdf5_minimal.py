"""A read-only reader for the one HDF5 layout the G4 step files use.

This is not an HDF5 implementation. It reads files written with the default
("earliest") libhdf5 layout — superblock version 0, old-style groups with a
symbol-table B-tree and a local heap, version-1 object headers — and returns
one-dimensional datasets of fixed-size numeric or compound type. Contiguous
and chunked storage are both handled; a filter pipeline (compression) is
refused rather than guessed at.

It exists because the G4 showers are stored as HDF5 and ``h5py`` cannot be
installed in every environment this code has to run in. Where ``h5py`` is
available it should be preferred, and :func:`read_dataset` gives the same
array either way; :mod:`lighthit.experimental.g4_source` picks whichever is
present. Everything here is read-only: nothing writes HDF5.

The parsed structures follow the HDF5 file format specification: the
superblock, symbol table entries, local heaps, version-1 B-tree nodes, symbol
table nodes, version-1 object headers and the dataspace, datatype and data
layout messages.
"""
from dataclasses import dataclass
import numpy as np

__all__ = ["File", "read_dataset", "list_datasets"]

SIGNATURE = b"\x89HDF\r\n\x1a\n"
_CLASS_FIXED, _CLASS_FLOAT, _CLASS_STRING, _CLASS_COMPOUND = 0, 1, 3, 6


def _unpack(buffer, offset, size):
    return int.from_bytes(buffer[offset:offset + size], "little")


@dataclass
class _Header:
    """The messages of one object header, keyed by message type."""
    messages: dict

    def one(self, kind):
        values = self.messages.get(kind, [])
        if len(values) != 1:
            raise ValueError(f"Expected exactly one message of type {kind}, got {len(values)}")
        return values[0]


class File:
    """An open HDF5 file, read through ``numpy.memmap``."""

    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        raw = memoryview(self.data)
        if bytes(raw[:8]) != SIGNATURE:
            raise ValueError("Not an HDF5 file")
        if raw[8] != 0:
            raise ValueError("Only superblock version 0 is supported here")
        self.raw = raw
        self.offset_size = raw[13]
        self.length_size = raw[14]
        root_entry = 24 + 4 * self.offset_size
        self.root = self._symbol_entry(root_entry)

    # -- primitive readers ------------------------------------------------

    def _offset(self, position):
        value = _unpack(self.raw, position, self.offset_size)
        undefined = (1 << (8 * self.offset_size)) - 1
        return None if value == undefined else value

    def _length(self, position):
        return _unpack(self.raw, position, self.length_size)

    def _symbol_entry(self, position):
        size = self.offset_size
        entry = {"name_offset": _unpack(self.raw, position, size),
                 "header": _unpack(self.raw, position + size, size),
                 "cache_type": _unpack(self.raw, position + 2 * size, 4)}
        scratch = position + 2 * size + 8
        if entry["cache_type"] == 1:
            entry["btree"] = _unpack(self.raw, scratch, size)
            entry["heap"] = _unpack(self.raw, scratch + size, size)
        return entry

    def _heap_name(self, heap_address, name_offset):
        if bytes(self.raw[heap_address:heap_address + 4]) != b"HEAP":
            raise ValueError("Local heap signature missing")
        data = self._offset(heap_address + 8 + 2 * self.length_size)
        start = data + name_offset
        end = start
        while self.raw[end] != 0:
            end += 1
        return bytes(self.raw[start:end]).decode("utf-8")

    # -- group walking ----------------------------------------------------

    def _symbol_nodes(self, btree_address):
        """Addresses of the symbol table nodes below a group B-tree."""
        if bytes(self.raw[btree_address:btree_address + 4]) != b"TREE":
            raise ValueError("B-tree signature missing")
        node_type = self.raw[btree_address + 4]
        level = self.raw[btree_address + 5]
        used = _unpack(self.raw, btree_address + 6, 2)
        if node_type != 0:
            raise ValueError("Expected a group B-tree")
        position = btree_address + 8 + 2 * self.offset_size
        children = []
        for index in range(used):
            position += self.length_size          # key
            children.append(self._offset(position))
            position += self.offset_size
        if level == 0:
            return children
        below = []
        for child in children:
            below.extend(self._symbol_nodes(child))
        return below

    def _group_links(self, entry):
        """Map names to object header addresses for one group."""
        links = {}
        for node in self._symbol_nodes(entry["btree"]):
            if bytes(self.raw[node:node + 4]) != b"SNOD":
                raise ValueError("Symbol table node signature missing")
            count = _unpack(self.raw, node + 6, 2)
            stride = 2 * self.offset_size + 24
            for index in range(count):
                child = self._symbol_entry(node + 8 + index * stride)
                name = self._heap_name(entry["heap"], child["name_offset"])
                links[name] = child
        return links

    def _object_header(self, address):
        version = self.raw[address]
        if version != 1:
            raise ValueError("Only version 1 object headers are supported here")
        count = _unpack(self.raw, address + 2, 2)
        messages = {}
        position = address + 16
        remaining = count
        continuations = []
        while remaining:
            kind = _unpack(self.raw, position, 2)
            size = _unpack(self.raw, position + 2, 2)
            body = position + 8
            if kind == 0x0010:                     # continuation
                continuations.append((self._offset(body),
                                      self._length(body + self.offset_size)))
            else:
                messages.setdefault(kind, []).append((body, size))
            position = body + size
            remaining -= 1
            if remaining and continuations and position >= address + 16 + self._header_span(address):
                start, _ = continuations.pop(0)
                position = start
        return _Header(messages)

    def _header_span(self, address):
        return _unpack(self.raw, address + 8, 4)

    # -- dataset reading --------------------------------------------------

    def _datatype(self, position):
        first = self.raw[position]
        version, kind = first >> 4, first & 0x0F
        bits = bytes(self.raw[position + 1:position + 4])
        size = _unpack(self.raw, position + 4, 4)
        properties = position + 8
        if kind == _CLASS_FIXED:
            signed = bool(bits[0] & 0x08)
            order = ">" if bits[0] & 0x01 else "<"
            return np.dtype(f"{order}{'i' if signed else 'u'}{size}"), size
        if kind == _CLASS_FLOAT:
            order = ">" if bits[0] & 0x01 else "<"
            return np.dtype(f"{order}f{size}"), size
        if kind == _CLASS_STRING:
            return np.dtype(f"S{size}"), size
        if kind != _CLASS_COMPOUND:
            raise ValueError(f"Unsupported datatype class {kind}")
        members = _unpack(bits, 0, 2)
        names, formats, offsets = [], [], []
        cursor = properties
        for _ in range(members):
            end = cursor
            while self.raw[end] != 0:
                end += 1
            name = bytes(self.raw[cursor:end]).decode("utf-8")
            if version == 1:
                cursor += (end - cursor + 8) // 8 * 8
                member_offset = _unpack(self.raw, cursor, 4)
                cursor += 4 + 1 + 3 + 4 + 4 + 16
            elif version in (2, 3):
                cursor = end + 1
                width = 1 if version == 3 else 4
                if version == 3:
                    width = max(1, (max(size, 1).bit_length() + 7) // 8)
                member_offset = _unpack(self.raw, cursor, width)
                cursor += width
                if version == 2:
                    cursor += 0
            else:
                raise ValueError(f"Unsupported compound datatype version {version}")
            member_type, member_size = self._datatype(cursor)
            cursor += self._datatype_span(cursor)
            names.append(name)
            formats.append(member_type)
            offsets.append(member_offset)
        return np.dtype({"names": names, "formats": formats,
                         "offsets": offsets, "itemsize": size}), size

    def _datatype_span(self, position):
        first = self.raw[position]
        version, kind = first >> 4, first & 0x0F
        size = _unpack(self.raw, position + 4, 4)
        if kind in (_CLASS_FIXED, _CLASS_STRING):
            return 8 + (4 if kind == _CLASS_FIXED else 0)
        if kind == _CLASS_FLOAT:
            return 8 + 12
        if kind != _CLASS_COMPOUND:
            raise ValueError(f"Unsupported datatype class {kind}")
        members = _unpack(bytes(self.raw[position + 1:position + 4]), 0, 2)
        cursor = position + 8
        for _ in range(members):
            end = cursor
            while self.raw[end] != 0:
                end += 1
            if version == 1:
                cursor += (end - cursor + 8) // 8 * 8
                cursor += 4 + 1 + 3 + 4 + 4 + 16
            else:
                cursor = end + 1
                width = max(1, (max(size, 1).bit_length() + 7) // 8) if version == 3 else 4
                cursor += width
            cursor += self._datatype_span(cursor)
        return cursor - position

    def _filters(self, position):
        """The filter pipeline, as a list of (id, client data)."""
        version = self.raw[position]
        count = self.raw[position + 1]
        filters = []
        if version == 1:
            cursor = position + 8
            for _ in range(count):
                identifier = _unpack(self.raw, cursor, 2)
                name_length = _unpack(self.raw, cursor + 2, 2)
                values = _unpack(self.raw, cursor + 6, 2)
                cursor += 8 + name_length
                data = [_unpack(self.raw, cursor + 4 * i, 4) for i in range(values)]
                cursor += 4 * values + (4 if values % 2 else 0)
                filters.append((identifier, data))
        elif version == 2:
            cursor = position + 2
            for _ in range(count):
                identifier = _unpack(self.raw, cursor, 2)
                cursor += 2
                name_length = 0 if identifier < 256 else _unpack(self.raw, cursor, 2)
                if identifier >= 256:
                    cursor += 2
                values = _unpack(self.raw, cursor + 2, 2)
                cursor += 4 + name_length
                data = [_unpack(self.raw, cursor + 4 * i, 4) for i in range(values)]
                cursor += 4 * values
                filters.append((identifier, data))
        else:
            raise ValueError("Unsupported filter pipeline message")
        return filters

    @staticmethod
    def _undo_filters(payload, filters, itemsize):
        """Reverse the pipeline for one chunk. Filters apply in reverse order."""
        import zlib
        for identifier, data in reversed(filters):
            if identifier == 1:                     # deflate
                payload = zlib.decompress(bytes(payload))
            elif identifier == 2:                   # shuffle
                width = data[0] if data else itemsize
                if width > 1:
                    block = np.frombuffer(payload, dtype=np.uint8)
                    count = len(block) // width
                    head = block[:count * width].reshape(width, count).T.reshape(-1)
                    payload = head.tobytes() + block[count * width:].tobytes()
            elif identifier == 3:                   # fletcher32 checksum
                payload = bytes(payload)[:-4]
            else:
                raise ValueError(f"Unsupported HDF5 filter {identifier}")
        return payload

    def _dataspace(self, position):
        version = self.raw[position]
        rank = self.raw[position + 1]
        if version != 1:
            raise ValueError("Only version 1 dataspace messages are supported here")
        dims = [self._length(position + 8 + index * self.length_size)
                for index in range(rank)]
        return tuple(dims)

    def _layout(self, position):
        version = self.raw[position]
        if version not in (1, 2, 3):
            raise ValueError("Unsupported data layout message")
        if version == 3:
            kind = self.raw[position + 1]
            body = position + 2
            if kind == 1:                                   # contiguous
                return {"class": "contiguous", "address": self._offset(body),
                        "size": self._length(body + self.offset_size)}
            if kind == 2:                                   # chunked
                rank = self.raw[body]
                address = self._offset(body + 1)
                dims = [_unpack(self.raw, body + 1 + self.offset_size + 4 * i, 4)
                        for i in range(rank)]
                return {"class": "chunked", "address": address, "dims": dims}
            if kind == 0:                                   # compact
                size = _unpack(self.raw, body, 2)
                return {"class": "compact", "address": body + 2, "size": size}
        raise ValueError("Only version 3 data layout messages are supported here")

    def _chunk_addresses(self, address, rank):
        """Leaf entries of a raw-data chunk B-tree: (offsets, size, address)."""
        if address is None:
            return []
        if bytes(self.raw[address:address + 4]) != b"TREE":
            raise ValueError("Chunk B-tree signature missing")
        if self.raw[address + 4] != 1:
            raise ValueError("Expected a raw-data chunk B-tree")
        level = self.raw[address + 5]
        used = _unpack(self.raw, address + 6, 2)
        position = address + 8 + 2 * self.offset_size
        key_size = 8 + 8 * rank
        entries = []
        for index in range(used):
            size = _unpack(self.raw, position, 4)
            filters = _unpack(self.raw, position + 4, 4)
            offsets = [_unpack(self.raw, position + 8 + 8 * i, 8) for i in range(rank)]
            child = self._offset(position + key_size)
            entries.append((offsets, size, filters, child))
            position += key_size + self.offset_size
        if level == 0:
            return [(offsets, size, child) for offsets, size, filters, child in entries]
        below = []
        for _, _, _, child in entries:
            below.extend(self._chunk_addresses(child, rank))
        return below

    def dataset(self, path):
        """Read one dataset by its slash-separated path."""
        entry = self.root
        parts = [part for part in path.strip("/").split("/") if part]
        for name in parts:
            links = self._group_links(entry)
            if name not in links:
                raise KeyError(f"{name} not found; have {sorted(links)}")
            entry = links[name]
            if "btree" not in entry and name != parts[-1]:
                raise KeyError(f"{name} is not a group")
        header = self._object_header(entry["header"])
        filters = self._filters(header.one(0x000B)[0]) if 0x000B in header.messages else []
        shape = self._dataspace(header.one(0x0001)[0])
        dtype, _ = self._datatype(header.one(0x0003)[0])
        layout = self._layout(header.one(0x0008)[0])
        count = int(np.prod(shape)) if shape else 1
        if layout["class"] in ("contiguous", "compact"):
            if filters:
                raise ValueError("A filtered contiguous dataset is not expected")
            start = layout["address"]
            block = np.frombuffer(self.data, dtype=dtype, count=count, offset=start)
            return block.reshape(shape)
        if len(shape) != 1:
            raise ValueError("Only one-dimensional chunked datasets are supported here")
        out = np.zeros(count, dtype=dtype)
        chunk = layout["dims"][0]
        for offsets, size, address in self._chunk_addresses(layout["address"], len(shape) + 1):
            start = offsets[0]
            length = min(chunk, count - start)
            if length <= 0:
                continue
            payload = memoryview(self.data)[address:address + size]
            if filters:
                payload = self._undo_filters(payload, filters, dtype.itemsize)
            piece = np.frombuffer(payload, dtype=dtype, count=length)
            out[start:start + length] = piece
        return out.reshape(shape)

    def groups(self, path=""):
        entry = self.root
        for name in [part for part in path.strip("/").split("/") if part]:
            entry = self._group_links(entry)[name]
        return sorted(self._group_links(entry))


def read_dataset(path, dataset):
    """Read one dataset from an HDF5 file, with ``h5py`` when it is installed."""
    try:
        import h5py
    except ImportError:
        return File(path).dataset(dataset)
    with h5py.File(path, "r") as handle:
        return handle[dataset][...]


def list_datasets(path, group=""):
    """Names directly under one group."""
    try:
        import h5py
    except ImportError:
        return File(path).groups(group)
    with h5py.File(path, "r") as handle:
        node = handle[group] if group else handle
        return sorted(node.keys())

"""Numerical receiver settings recorded with prepared route data."""
from dataclasses import asdict,is_dataclass
from pathlib import Path
import os
from .receiver_vertex import ReceiverVertexConfig


def decode_settings(values,base):
    settings=dict(values or {})
    vertex=settings.get('receiver_vertex')
    if vertex is not None:settings['receiver_vertex']=ReceiverVertexConfig(**vertex)
    path=settings.get('first_entry_tables')
    if path is not None:settings['first_entry_tables']=(Path(base)/path).resolve()
    return settings


def encode_settings(values,base):
    settings=dict(values or {})
    vertex=settings.get('receiver_vertex')
    if is_dataclass(vertex):settings['receiver_vertex']=asdict(vertex)
    path=settings.get('first_entry_tables')
    if path is not None:settings['first_entry_tables']=os.path.relpath(Path(path).resolve(),Path(base).resolve())
    return settings

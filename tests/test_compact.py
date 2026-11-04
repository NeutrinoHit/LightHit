"""Exact and reduced transports of a runtime bundle, unpack and file checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from lighthit import compact


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bundle(root):
    """Small runtime: a smooth CDF table, an RTE-like npz with two orders, a plain file."""
    rng = np.random.default_rng(3)
    u = np.linspace(0., 1., 64)
    rate = rng.uniform(.5, 3., (5, 3, 1, 2))
    cdf = 1. - np.exp(-rate * u[None, None, :, None])
    (root / 'tables/cdf').mkdir(parents=True)
    np.save(root / 'tables/cdf/hit_cdf.npy', cdf)
    coefficients = (rng.normal(size=(6, 9, 2)) + 1j * rng.normal(size=(6, 9, 2))) * 10. ** rng.uniform(-6, 0, (6, 1, 1))
    (root / 'tables/rte-cache').mkdir(parents=True)
    np.savez(root / 'tables/rte-cache/directional-a.npz', coefficients=coefficients, omega=np.arange(9.))
    (root / 'tables/notes.json').write_text('{"kept": true}\n')
    np.save(root / 'tables/big.npy', np.arange(300000, dtype=float))
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            rel = str(path.relative_to(root))
            files[rel] = dict(size=path.stat().st_size, sha256=_digest(path))
    (root / 'bundle.json').write_text(json.dumps(dict(complete=True, files=files)))
    return cdf, coefficients


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.setattr(compact, 'DERIVED', {})
    monkeypatch.setattr(compact, 'GROUPS', [
        ('cdf', 'tables/cdf/hit_cdf.npy', None, 2, 'pod', 'cdf'),
        ('rte-cache', 'tables/rte-cache/directional-*.npz', ('coefficients',), 0, 'scaled32', 'field')])


def _pack(source, output, *extra):
    return compact.main(['pack', '--bundle', str(source), '--output', str(output), '--deflate', '--jobs', '1', *extra])


def _unpack(transport, output):
    return compact.main(['unpack', '--input', str(transport), '--output', str(output), '--jobs', '1'])


def test_shuffled_deflate_is_bit_exact(tmp_path):
    a = np.random.default_rng(0).normal(size=(37, 11)).astype(np.float32)
    compact.shz_write(tmp_path / 'a.shz', a)
    assert np.array_equal(compact.shz_read(tmp_path / 'a.shz'), a)


def test_exact_transport_restores_every_file(tmp_path, groups):
    source = tmp_path / 'full'; source.mkdir(); _bundle(source)
    assert _pack(source, tmp_path / 'exact') == 0
    manifest = json.loads((tmp_path / 'exact/compact.json').read_text())
    assert manifest['form'] == 'exact' and manifest['tolerance'] is None
    assert _unpack(tmp_path / 'exact', tmp_path / 'runtime') == 0
    report = json.loads((tmp_path / 'runtime/unpack-report.json').read_text())
    assert report['form'] == 'exact' and not report['differing']
    assert compact.main(['verify', '--bundle', str(tmp_path / 'runtime')]) == 0


def test_reduced_transport_keeps_row_error_and_marks_order1(tmp_path, groups):
    source = tmp_path / 'full'; source.mkdir(); cdf, coefficients = _bundle(source)
    assert _pack(source, tmp_path / 'reduced', '--tolerance', '1e-6', '--drop-rte-order1') == 0
    assert _unpack(tmp_path / 'reduced', tmp_path / 'runtime') == 0
    got = np.load(tmp_path / 'runtime/tables/cdf/hit_cdf.npy')
    rows, back = compact._rows(cdf, 2), compact._rows(got, 2)
    assert np.max(np.abs(back - rows) / np.abs(rows).max(axis=1)[:, None]) <= 1e-6
    assert np.array_equal(back[:, 0], rows[:, 0]) and np.array_equal(back[:, -1], rows[:, -1])
    assert np.all(np.diff(back, axis=1) >= 0)
    with np.load(tmp_path / 'runtime/tables/rte-cache/directional-a.npz') as d:
        restored, omega = d['coefficients'], d['omega']
    assert np.isnan(restored[..., 0]).all() and np.array_equal(omega, np.arange(9.))
    scale = np.abs(coefficients[..., 1]).max(axis=1, keepdims=True)
    assert np.max(np.abs(restored[..., 1] - coefficients[..., 1]) / scale) <= 1e-6
    report = json.loads((tmp_path / 'runtime/unpack-report.json').read_text())
    assert report['form'] == 'reduced' and report['drop_rte_order1']
    assert compact.main(['verify', '--bundle', str(tmp_path / 'runtime')]) == 0


def test_verify_and_unpack_refuse_changed_exact_files(tmp_path, groups):
    source = tmp_path / 'full'; source.mkdir(); _bundle(source)
    assert compact.main(['verify', '--bundle', str(source)]) == 0
    (source / 'tables/notes.json').write_text('{"kept": false}\n')
    assert compact.main(['verify', '--bundle', str(source)]) == 2
    (source / 'tables/notes.json').write_text('{"kept": true}\n')
    _pack(source, tmp_path / 'exact')
    (tmp_path / 'exact/files/tables/notes.json').write_text('{"kept": 0}\n')
    with pytest.raises(ValueError, match='transport files missing or changed'):
        _unpack(tmp_path / 'exact', tmp_path / 'runtime')

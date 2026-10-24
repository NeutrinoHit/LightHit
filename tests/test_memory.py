"""Operating-system headroom rather than a fraction of installed RAM."""
from pathlib import Path

from lighthit import _memory


def test_macos_headroom_uses_page_size_and_reclaimable_pages(monkeypatch):
    monkeypatch.setattr(_memory.sys, 'platform', 'darwin')
    report = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
              'Pages free: 10.\nPages inactive: 20.\nPages speculative: 3.\n'
              'Pages wired down: 999999.\n')
    monkeypatch.setattr(_memory.subprocess, 'check_output', lambda *a, **kw: report)
    assert _memory.available_bytes() == 33 * 16384


def test_linux_headroom_respects_parent_cgroup_limit(monkeypatch):
    monkeypatch.setattr(_memory.sys, 'platform', 'linux')
    reports = {'/proc/meminfo': 'MemTotal: 1000000 kB\nMemAvailable: 1000 kB\n',
               '/proc/self/cgroup': '0::/parent/child\n',
               '/sys/fs/cgroup/parent/child/memory.max': 'max',
               '/sys/fs/cgroup/parent/child/memory.current': '10',
               '/sys/fs/cgroup/parent/memory.max': '50000',
               '/sys/fs/cgroup/parent/memory.current': '40000',
               '/sys/fs/cgroup/memory.max': 'max',
               '/sys/fs/cgroup/memory.current': '90000'}
    def read(path):
        if str(path) not in reports:
            raise FileNotFoundError(path)
        return reports[str(path)]
    monkeypatch.setattr(Path, 'read_text', read)
    assert _memory.available_bytes() == 10000

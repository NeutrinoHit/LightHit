"""Available allocation headroom reported by the operating system."""
import os
from pathlib import Path
import re
import subprocess
import sys


def available_bytes():
    """Reclaimable/free RAM, respecting a Linux cgroup's memory limit.

    This is a snapshot, not a reservation. Failure is explicit so a caller
    can request a fixed budget instead of silently assuming a machine size.
    """
    if sys.platform == 'darwin':
        report = subprocess.check_output(['vm_stat'], text=True)
        size = re.search(r'page size of (\d+) bytes', report)
        pages = dict((name, int(count)) for name, count in
                     re.findall(r'^([^:\n]+):\s+(\d+)\.', report, re.M))
        names = ('Pages free', 'Pages inactive', 'Pages speculative')
        if size is None or any(name not in pages for name in names):
            raise RuntimeError('cannot read available memory; set grouped_scratch_bytes')
        return int(size[1]) * sum(pages[name] for name in names)
    if sys.platform.startswith('linux'):
        values = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            name, value = line.split(':', 1)
            values[name] = int(value.split()[0]) * 1024
        available = values['MemAvailable']
        # cgroup paths are relative to the current cgroup namespace. Read
        # ancestors too: a parent may impose the tighter allocation limit.
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, relative = line.split(':', 2)
            if hierarchy != '0' or controllers:
                continue
            root = Path('/sys/fs/cgroup')
            directory = root / relative.lstrip('/')
            while directory == root or root in directory.parents:
                try:
                    limit = (directory / 'memory.max').read_text().strip()
                    used = int((directory / 'memory.current').read_text())
                    if limit != 'max':
                        available = min(available, max(0, int(limit) - used))
                except FileNotFoundError:
                    pass
                if directory == root:
                    break
                directory = directory.parent
        return available
    try:
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    except (AttributeError, ValueError, OSError) as error:
        raise RuntimeError('cannot read available memory; set grouped_scratch_bytes') from error

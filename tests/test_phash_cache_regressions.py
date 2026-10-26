from __future__ import annotations

import os
import multiprocessing
from types import SimpleNamespace

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import pytest
from PIL import Image

from modimg import phash
from modimg.engines.phash_allow import PHashAllowlistEngine
from modimg.engines.phash_block import PHashBlocklistEngine
from modimg.enums import EngineStatus
from modimg.phash import append_phash_to_allowlist, frame_phash_hex_int, load_phash_exact_map, load_phash_list
from modimg.types import Frame


@pytest.mark.parametrize("engine_class", [PHashAllowlistEngine, PHashBlocklistEngine])
@pytest.mark.parametrize("distance", [0, 1])
def test_phash_engine_reloads_same_size_replacement_with_preserved_timestamp(tmp_path, engine_class, distance) -> None:
    path = tmp_path / "list.txt"
    frame = Frame(0, Image.new("RGB", (8, 8), color=(10, 20, 30)))
    try:
        old_hash, _ = frame_phash_hex_int(frame)
        new_hash = f"{int(old_hash, 16) ^ 0xffffffffffffffff:016x}"
        path.write_text(f"{old_hash},known\n", encoding="utf-8")
        engine = engine_class(str(path), max_distance=distance)
        score = "phash_allow_match" if engine_class is PHashAllowlistEngine else "phash_block_match"

        before = engine.execute("image.png", [frame])
        assert before.status == EngineStatus.OK
        assert before.scores[score] == 1.0
        original_stat = path.stat()
        replacement = tmp_path / "replacement.txt"
        replacement.write_text(f"{new_hash},known\n", encoding="utf-8")
        os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        replacement.replace(path)

        after = engine.execute("image.png", [frame])

        assert after.status == EngineStatus.OK
        assert after.scores[score] == 0.0
    finally:
        frame.close()


@pytest.mark.parametrize("loader", [load_phash_list, load_phash_exact_map])
def test_phash_cache_honors_each_callers_default_label(tmp_path, loader) -> None:
    path = tmp_path / "unlabeled.txt"
    path.write_text("0123456789abcdef\n", encoding="utf-8")

    loader(str(path), default_label="allow")
    result = loader(str(path), default_label="block")

    if loader is load_phash_list:
        assert result == [("0123456789abcdef", "block", 0x0123456789abcdef, 16)]
    else:
        assert result == {16: {0x0123456789abcdef: ("0123456789abcdef", "block")}}


@pytest.mark.parametrize("loader", [load_phash_list, load_phash_exact_map])
@pytest.mark.parametrize(
    ("setting", "value", "expected_count"),
    [
        ("PHASH_LIST_MAX_ENTRIES", "1", 1),
        ("PHASH_MAX_HEX_LENGTH", "8", 0),
        ("PHASH_LIST_MAX_BYTES", "1", 0),
    ],
)
def test_phash_cache_obeys_changed_parser_limits(monkeypatch, tmp_path, loader, setting, value, expected_count) -> None:
    path = tmp_path / "limited.txt"
    path.write_text("0123456789abcdef,one\nfedcba9876543210,two\n", encoding="utf-8")
    monkeypatch.setenv("PHASH_LIST_MAX_ENTRIES", "100000")
    monkeypatch.setenv("PHASH_MAX_HEX_LENGTH", "256")
    monkeypatch.setenv("PHASH_LIST_MAX_BYTES", "10000000")
    loader(str(path), default_label="allow")

    monkeypatch.setenv(setting, value)
    result = loader(str(path), default_label="allow")

    count = len(result) if loader is load_phash_list else sum(len(entries) for entries in result.values())
    assert count == expected_count


def _append_in_process(path, hash_value, barrier, results):
    barrier.wait(timeout=10)
    results.put((hash_value, append_phash_to_allowlist(hash_value, path, "new")))


def _controlled_append_in_process(path, hash_value, first, first_ready, release_first, second_progress, results):
    atomic_write = phash.atomic_write_text

    def controlled_write(*args, **kwargs):
        if first:
            first_ready.set()
            assert release_first.wait(timeout=10)
        else:
            second_progress.set()
        return atomic_write(*args, **kwargs)

    phash.atomic_write_text = controlled_write
    if not first:
        if os.name == "nt":
            locking = msvcrt.locking

            def observed_lock(fd, mode, count):
                if mode == msvcrt.LK_NBLCK:
                    second_progress.set()
                return locking(fd, mode, count)

            msvcrt.locking = observed_lock
        else:
            flock = fcntl.flock

            def observed_lock(fd, mode):
                if mode == (fcntl.LOCK_EX | fcntl.LOCK_NB):
                    second_progress.set()
                return flock(fd, mode)

            fcntl.flock = observed_lock
    results.put((hash_value, phash.append_phash_to_allowlist(hash_value, path, "new")))


def _process_appends(path, hashes):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(hashes))
    results = context.Queue()
    processes = [context.Process(target=_append_in_process, args=(str(path), value, barrier, results)) for value in hashes]
    try:
        for process in processes:
            process.start()
        values = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        return values
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.parametrize("duplicate", [False, True])
def test_phash_append_serializes_process_transactions(monkeypatch, tmp_path, duplicate) -> None:
    monkeypatch.setenv("PHASH_LIST_MAX_BYTES", "10000000")
    path = tmp_path / "shared.txt"
    existing = "0123456789abcdef,existing\n"
    path.write_text(existing, encoding="utf-8")
    path.chmod(0o640)
    hashes = ["f000000000000001", "f000000000000001" if duplicate else "f000000000000002"]
    context = multiprocessing.get_context("spawn")
    first_ready, release_first, second_progress = (context.Event() for _ in range(3))
    queue = context.Queue()
    processes = [
        context.Process(
            target=_controlled_append_in_process,
            args=(str(path), value, index == 0, first_ready, release_first, second_progress, queue),
        )
        for index, value in enumerate(hashes)
    ]
    try:
        processes[0].start()
        assert first_ready.wait(timeout=10)
        processes[1].start()
        assert second_progress.wait(timeout=10)
        release_first.set()
        results = [queue.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        release_first.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        queue.close()
        queue.join_thread()

    added = [line for line in path.read_text(encoding="utf-8").splitlines() if line.endswith(",new")]
    assert sorted(added) == sorted(f"{value},new" for value in set(hashes))
    assert sum(success for _, success in results) == len(set(hashes))
    assert path.read_text(encoding="utf-8").startswith(existing)
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("failure", ["write", "duplicate", "limit"])
def test_phash_append_releases_process_lock_after_early_exit(monkeypatch, tmp_path, failure) -> None:
    path = tmp_path / "failed.txt"
    original = "0123456789abcdef,existing\n"
    path.write_text(original, encoding="utf-8")

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        value = "f000000000000001"
        if failure == "write":
            patch.setattr("modimg.phash.atomic_write_text", fail_write)
        elif failure == "duplicate":
            value = "0123456789abcdef"
        else:
            patch.setenv("PHASH_LIST_MAX_BYTES", "1")
        assert append_phash_to_allowlist(value, str(path), "new") is False
    assert path.read_text(encoding="utf-8") == original

    results = _process_appends(path, ["f000000000000001"])

    assert results == [("f000000000000001", True)]
    assert path.read_text(encoding="utf-8") == original + "f000000000000001,new\n"


@pytest.mark.parametrize("failure", [None, "acquire", "body", "unlock"])
def test_windows_phash_lock_releases_file_handle_on_each_exit(monkeypatch, tmp_path, failure) -> None:
    calls = []

    def locking(fd, mode, count):
        calls.append((fd, mode, count, os.lseek(fd, 0, os.SEEK_CUR)))
        if (failure == "acquire" and mode == 1) or (failure == "unlock" and mode == 2):
            raise OSError("lock failure")

    monkeypatch.setattr(phash, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(phash, "msvcrt", SimpleNamespace(locking=locking, LK_LOCK=1, LK_NBLCK=1, LK_UNLCK=2), raising=False)
    path = str(tmp_path / "windows.txt")

    def locked_operation():
        with phash._phash_file_lock(path):
            if failure == "body":
                raise RuntimeError("body failure")

    if failure:
        with pytest.raises(RuntimeError if failure == "body" else OSError):
            locked_operation()
    else:
        locked_operation()

    assert [mode for _, mode, _, _ in calls] == ([1] if failure == "acquire" else [1, 2])
    assert all(count == 1 and position == 0 for _, _, count, position in calls)
    with pytest.raises(OSError):
        os.fstat(calls[0][0])


def test_phash_process_lock_is_released_when_writer_dies(tmp_path) -> None:
    path = tmp_path / "terminated.txt"
    path.write_text("0123456789abcdef,existing\n", encoding="utf-8")
    context = multiprocessing.get_context("spawn")
    first_ready, release_first, second_progress = (context.Event() for _ in range(3))
    queue = context.Queue()
    process = context.Process(
        target=_controlled_append_in_process,
        args=(str(path), "f000000000000001", True, first_ready, release_first, second_progress, queue),
    )
    try:
        process.start()
        assert first_ready.wait(timeout=10)
        process.terminate()
        process.join(timeout=10)
        assert not process.is_alive()

        assert _process_appends(path, ["f000000000000002"]) == [("f000000000000002", True)]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()


def test_phash_lock_sidecar_keeps_its_inode_across_transactions(tmp_path) -> None:
    path = tmp_path / "persistent.txt"
    assert append_phash_to_allowlist("f000000000000001", str(path), "new") is True
    lock_path = tmp_path / "persistent.txt.lock"
    first_inode = lock_path.stat().st_ino

    assert append_phash_to_allowlist("f000000000000002", str(path), "new") is True

    assert lock_path.stat().st_ino == first_inode


def test_windows_phash_lock_never_accesses_locked_bytes(monkeypatch, tmp_path) -> None:
    """Windows denies data I/O even before a contender calls locking()."""
    real_open = open
    handles = []

    class LockHandle:
        def __init__(self, *args, **kwargs):
            self.file = real_open(*args, **kwargs)
            handles.append(self.file)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file.close()

        def fileno(self):
            return self.file.fileno()

        def seek(self, *args):
            return self.file.seek(*args)

        def read(self, *args):
            raise PermissionError("another process owns the byte range")

        write = read

    calls = []
    monkeypatch.setattr(phash, "open", LockHandle, raising=False)
    monkeypatch.setattr(phash, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(phash, "msvcrt", SimpleNamespace(locking=lambda *args: calls.append(args), LK_LOCK=1, LK_NBLCK=1, LK_UNLCK=2), raising=False)

    with phash._phash_file_lock(str(tmp_path / "contended.txt")):
        assert [mode for _, mode, _ in calls] == [1]
    assert [mode for _, mode, _ in calls] == [1, 2]
    assert all(handle.closed for handle in handles)
    assert (tmp_path / "contended.txt.lock").stat().st_size == 0


@pytest.mark.parametrize("windows", [True, False])
def test_phash_lock_contention_has_bounded_wait(monkeypatch, tmp_path, windows) -> None:
    import errno

    clock = [0.0]
    calls = []

    def contended(fd, mode, *args):
        calls.append(fd)
        raise OSError(errno.EACCES if windows else errno.EAGAIN, "busy")

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(phash, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep), raising=False)
    monkeypatch.setattr(phash, "os", SimpleNamespace(name="nt" if windows else "posix"))
    monkeypatch.setattr(phash, "msvcrt", SimpleNamespace(locking=contended, LK_LOCK=1, LK_NBLCK=1, LK_UNLCK=2), raising=False)
    monkeypatch.setattr(phash, "fcntl", SimpleNamespace(flock=contended, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8), raising=False)
    with pytest.raises(TimeoutError, match="pHash"):
        with phash._phash_file_lock(str(tmp_path / "busy.txt")):
            pytest.fail("transaction must not run without acquiring lock")
    assert 0 < clock[0] <= 10.1
    assert 1 < len(calls) < 205
    with pytest.raises(OSError):
        os.fstat(calls[0])


@pytest.mark.parametrize("windows", [True, False])
def test_phash_lock_recovers_from_transient_contention(monkeypatch, tmp_path, windows) -> None:
    import errno

    calls = []

    def transient(fd, mode, *args):
        calls.append(mode)
        if len(calls) == 1:
            raise OSError(errno.EACCES if windows else errno.EINTR, "retry")

    monkeypatch.setattr(phash, "time", SimpleNamespace(monotonic=lambda: 0.0, sleep=lambda _: None), raising=False)
    monkeypatch.setattr(phash, "os", SimpleNamespace(name="nt" if windows else "posix"))
    monkeypatch.setattr(phash, "msvcrt", SimpleNamespace(locking=transient, LK_LOCK=1, LK_NBLCK=1, LK_UNLCK=2), raising=False)
    monkeypatch.setattr(phash, "fcntl", SimpleNamespace(flock=transient, LOCK_EX=2, LOCK_NB=4, LOCK_UN=8), raising=False)
    with phash._phash_file_lock(str(tmp_path / "retry.txt")):
        assert len(calls) == 2
    assert len(calls) == 3


@pytest.mark.parametrize("duplicate", [False, True])
def test_many_process_writes_preserve_existing_entries(tmp_path, duplicate) -> None:
    path = tmp_path / "many.txt"
    original = "# keep comment\n0123456789abcdef,existing"
    path.write_text(original, encoding="utf-8")
    hashes = [f"f00000000000000{1 if duplicate else i:x}" for i in range(1, 5)]
    results = _process_appends(path, hashes)
    assert sum(success for _, success in results) == len(set(hashes))
    assert path.read_text(encoding="utf-8").startswith(original + "\n")
    assert sorted(path.read_text(encoding="utf-8").splitlines()[2:]) == sorted(f"{hx},new" for hx in set(hashes))


@pytest.mark.parametrize("failure", ["acquire", "unlock"])
def test_failed_os_lock_does_not_block_later_process(monkeypatch, tmp_path, failure) -> None:
    import errno

    path = tmp_path / "os-failure.txt"
    original = "0123456789abcdef,existing\n"
    path.write_text(original, encoding="utf-8")
    if os.name == "nt":
        module, name, unlock = msvcrt, "locking", msvcrt.LK_UNLCK
    else:
        module, name, unlock = fcntl, "flock", fcntl.LOCK_UN
    real_lock = getattr(module, name)

    def fail_lock(fd, mode, *args):
        if (mode == unlock) == (failure == "unlock"):
            raise OSError(errno.EBADF, "injected lock failure")
        return real_lock(fd, mode, *args)

    with monkeypatch.context() as patch:
        patch.setattr(module, name, fail_lock)
        assert not append_phash_to_allowlist("f000000000000001", str(path), "new")
    expected = original + ("f000000000000001,new\n" if failure == "unlock" else "")
    assert path.read_text(encoding="utf-8") == expected
    assert _process_appends(path, ["f000000000000002"]) == [("f000000000000002", True)]
    assert path.read_text(encoding="utf-8") == expected + "f000000000000002,new\n"


def test_thread_writers_keep_distinct_hashes(tmp_path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    path = tmp_path / "threads.txt"
    path.write_text("# preserve\n", encoding="utf-8")
    barrier = Barrier(8)

    def append(index):
        barrier.wait(timeout=5)
        return append_phash_to_allowlist(f"{index:016x}", str(path), "new")

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(append, range(8)))
    assert sorted(path.read_text(encoding="utf-8").splitlines()) == sorted(["# preserve"] + [f"{i:016x},new" for i in range(8)])

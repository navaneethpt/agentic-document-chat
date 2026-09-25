"""One temporary Chroma collection per session, with operation-safe expiry."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import logging
from threading import Event, RLock, Thread
import time
from uuid import uuid4

logger = logging.getLogger(__name__)


class SessionExpired(RuntimeError):
    pass


class SessionBusy(RuntimeError):
    pass


@dataclass
class Library:
    id: str
    collection: object
    last_used: float
    documents: dict = field(default_factory=dict)
    active: int = 0
    healthy: bool = True
    lock: RLock = field(default_factory=RLock)
    messages: list = field(default_factory=list)
    latest_operation: dict | None = None


class SessionManager:
    def __init__(self, client, embedding_function, *, ttl=3600, clock=time.monotonic,
                 start_worker=True):
        self.client = client
        self.embedding_function = embedding_function
        self.ttl = ttl
        self.clock = clock
        self._libraries: dict[str, Library] = {}
        self._lock = RLock()
        self._stop = Event()
        self._worker = None
        if start_worker:
            self._worker = Thread(target=self._cleanup_loop, daemon=True, name="rag-session-cleanup")
            self._worker.start()

    def ensure(self, session_id: str | None) -> tuple[Library, bool]:
        """Touch an existing session or replace an expired/missing one."""
        with self._lock:
            library = self._libraries.get(session_id)
            if library and (library.active or self.clock() - library.last_used < self.ttl):
                library.last_used = self.clock()
                return library, False
            if library:
                self._delete(library)
            identity = uuid4().hex
            collection = self.client.create_collection(
                name=f"session-{identity}", embedding_function=self.embedding_function,
                configuration={"hnsw": {"space": "cosine"}},
            )
            library = Library(identity, collection, self.clock())
            self._libraries[identity] = library
            return library, True

    def _existing(self, session_id: str) -> Library:
        library = self._libraries.get(session_id)
        if library is None:
            raise SessionExpired("This session expired. Start a new session and upload your documents again.")
        if not library.active and self.clock() - library.last_used >= self.ttl:
            self._delete(library)
            raise SessionExpired("This session expired. Start a new session and upload your documents again.")
        return library

    def snapshot(self, session_id: str) -> dict:
        """Read without touching the idle timer, including while a worker is busy."""
        with self._lock:
            library = self._existing(session_id)
            return {
                "id": library.id, "healthy": library.healthy,
                "documents": [{"id": key, "filename": doc.filename, "chunks": doc.chunks}
                              for key, doc in library.documents.copy().items()],
                "messages": deepcopy(library.messages),
                "operation": deepcopy(library.latest_operation),
            }

    def begin(self, session_id: str, kind: str, question: str = "") -> Library:
        """Reserve before dispatching work; concurrent requests fail rather than queue."""
        with self._lock:
            library = self._existing(session_id)
            if library.active:
                raise SessionBusy("Please wait for the current operation to finish.")
            library.active = 1
            library.latest_operation = {
                "id": uuid4().hex, "kind": kind, "status": "running",
                "question": question, "events": [], "error": None,
            }
            return library

    def record_event(self, library: Library, event: dict) -> None:
        with self._lock:
            events = library.latest_operation["events"]
            if len(events) < 128:
                events.append(deepcopy(event))

    def finish(self, library: Library, *, messages: list | None = None, error: str | None = None) -> None:
        with self._lock:
            if messages:
                library.messages.extend(messages)
            library.latest_operation["status"] = "failed" if error else "complete"
            library.latest_operation["error"] = error
            library.active = 0
            library.last_used = self.clock()

    @contextmanager
    def operation(self, session_id: str):
        with self._lock:
            library = self._libraries.get(session_id)
            if library is None:
                raise SessionExpired("This session expired. Refresh and upload your documents again.")
            if not library.active and self.clock() - library.last_used >= self.ttl:
                self._delete(library)
                raise SessionExpired("This session expired. Refresh and upload your documents again.")
            library.active += 1
        try:
            with library.lock:
                yield library
        finally:
            with self._lock:
                library.active -= 1
                library.last_used = self.clock()

    def clear(self, session_id: str) -> None:
        with self._lock:
            library = self._libraries.get(session_id)
            if library:
                if library.active:
                    raise SessionBusy("Please wait for the current operation before clearing the session.")
                self._delete(library)

    def _delete(self, library: Library) -> None:
        # Remove bookkeeping only after Chroma confirms deletion; failures are retried.
        self.client.delete_collection(library.collection.name)
        library.documents.clear()
        library.messages.clear()
        library.latest_operation = None
        del self._libraries[library.id]

    def cleanup(self) -> int:
        with self._lock:
            expired = [library for library in self._libraries.values()
                       if not library.active and self.clock() - library.last_used >= self.ttl]
            removed = 0
            for library in expired:
                try:
                    self._delete(library)
                    removed += 1
                except Exception:
                    logger.warning("Temporary library cleanup failed; will retry.")
            return removed

    def _cleanup_loop(self):
        while not self._stop.wait(60):
            self.cleanup()

    def close(self):
        """Stop the cleanup worker (primarily for tests and orderly shutdown)."""
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=2)

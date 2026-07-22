import time
from contextlib import contextmanager


class TimingNode:
    def __init__(self, name):
        self.name = str(name)
        self.elapsed = 0.0
        self.children = []
        self.details = []

    @contextmanager
    def section(self, name):
        child = TimingNode(name)
        self.children.append(child)
        start = time.perf_counter()
        try:
            yield child
        finally:
            child.elapsed += time.perf_counter() - start

    def add_detail(self, text):
        self.details.append(str(text))

    def add_child(self, name, elapsed=0.0, details=None):
        child = TimingNode(name)
        child.elapsed = float(elapsed)
        if details:
            if isinstance(details, (list, tuple)):
                child.details.extend(str(detail) for detail in details)
            else:
                child.details.append(str(details))
        self.children.append(child)
        return child

    def add_elapsed(self, name, elapsed):
        for child in self.children:
            if child.name == name:
                child.elapsed += float(elapsed)
                return child
        return self.add_child(name, elapsed)

    def format_lines(self, step=None):
        first = f"TIME {step} | TOTAL {self.elapsed:.3f}s" if step is not None else f"TIME | TOTAL {self.elapsed:.3f}s"
        lines = [first]
        for child in self.children:
            lines.extend(child._format_child(indent=1))
        return lines

    def _format_child(self, indent):
        prefix = "  " * indent
        detail = "" if not self.details else " | " + " | ".join(self.details)
        lines = [f"{prefix}{self.name:<24} {self.elapsed:.3f}s{detail}"]
        for child in self.children:
            lines.extend(child._format_child(indent + 1))
        return lines

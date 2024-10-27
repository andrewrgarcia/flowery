import sys
import os
from contextlib import contextmanager
import functools

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def flowery(verbose=False):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if not verbose:
                with suppress_stdout():
                    return method(self, *args, **kwargs)
            else:
                return method(self, *args, **kwargs)
        return wrapper
    return decorator


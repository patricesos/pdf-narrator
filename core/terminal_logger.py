class TermLogger:
    """Redirects stdout/stderr to a Tkinter widget."""

    def __init__(self, write_callback):
        self.write_callback = write_callback
        self.is_logging = False

    def write(self, message):
        if self.is_logging:
            return
        self.is_logging = True
        try:
            if message.strip():
                self.write_callback(message)
        finally:
            self.is_logging = False

    def flush(self):
        pass

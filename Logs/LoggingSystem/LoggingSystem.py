import sys


class LoggingSystem:
    """Записывает вывод одновременно в консоль и в файл."""

    def __init__(self, filename: str, mode: str = "w"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

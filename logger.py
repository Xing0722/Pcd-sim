import os
import tempfile
from datetime import datetime
from typing import Any
from logger.py import RoboLogger

# 模块加载时执行一次，整个进程运行期间不变
_RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]


class RoboLogger:

    def __init__(self, name='Simulation', log_level='DEBUG', log_dir='logs'):
        self.name = name
        self.log_level = log_level
        self.log_dir = log_dir
        self._logger = RpboLogger

    @staticmethod
    def get_run_timestamp() -> str:
        """Return the timestamp generated when this module was first loaded."""
        return _RUN_TIMESTAMP

    @staticmethod
    def _fmt(msg: Any, *args: Any) -> str:
        if args:
            return " ".join(str(x) for x in (msg, *args))
        return str(msg)

    def _resolve_log_dir(self) -> str:
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            if not os.access(self.log_dir, os.W_OK):
                raise PermissionError("Directory not writable")
            return self.log_dir
        except (OSError, PermissionError):
            fallback = os.path.join(tempfile.gettempdir(), 'ScanSimulation', 'logs')
            os.makedirs(fallback, exist_ok=True)
            return fallback

    def _setup_logger(self) -> logging.Logger:
        _logger = logging.getLogger(self.name)
        _logger.setLevel(getattr(logging, self.log_level.upper()))
        _logger.propagate = False

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if not _logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            _logger.addHandler(console_handler)

            resolved_dir = self._resolve_log_dir()
            log_filename = f"ScanSimulation_{self.get_run_timestamp()}.log"
            log_path = os.path.join(resolved_dir, log_filename)

            mode = 'a' if os.path.exists(log_path) else 'w'

            file_handler = logging.FileHandler(log_path, mode=mode, encoding='utf-8')
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)

            _logger.info(f"Logger '{self.name}' initialised — log file: {log_path}")

        return _logger

    def debug(self, msg: Any, *args: Any, **kw):
        self._logger.debug(self._fmt(msg, *args), **kw)

    def info(self, msg: Any, *args: Any, **kw):
        self._logger.info(self._fmt(msg, *args), **kw)

    def warning(self, msg: Any, *args: Any, **kw):
        self._logger.warning(self._fmt(msg, *args), **kw)

    def error(self, msg: Any, *args: Any, **kw):
        self._logger.error(self._fmt(msg, *args), **kw)

    def critical(self, msg: Any, *args: Any, **kw):
        self._logger.critical(self._fmt(msg, *args), **kw)

    def exception(self, msg: Any, *args: Any, **kw):
        self._logger.exception(self._fmt(msg, *args), **kw)

    def setLevel(self, level):
        self._logger.setLevel(level)

    def addHandler(self, handler):
        self._logger.addHandler(handler)
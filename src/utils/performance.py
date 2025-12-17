"""
Performance timing utilities for tracking execution time of different stages.
"""

import time
import logging
from typing import Optional, Callable, Any
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Context manager for timing code blocks."""

    def __init__(self, stage_name: str, log_level: int = logging.INFO):
        """
        Initialize timer.

        Args:
            stage_name: Name of the stage being timed
            log_level: Logging level for timing output
        """
        self.stage_name = stage_name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        logger.log(self.log_level, f"[PERF] Starting: {self.stage_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log results."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        if exc_type is None:
            # Success
            logger.log(
                self.log_level,
                f"[PERF] Completed: {self.stage_name} - {self.duration:.3f}s"
            )
        else:
            # Exception occurred
            logger.error(
                f"[PERF] Failed: {self.stage_name} - {self.duration:.3f}s - {exc_type.__name__}"
            )

        return False  # Don't suppress exceptions


def timed_operation(stage_name: Optional[str] = None):
    """
    Decorator to time function execution.

    Args:
        stage_name: Optional name for the stage (defaults to function name)

    Usage:
        @timed_operation("Custom Stage Name")
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            name = stage_name or func.__name__
            with PerformanceTimer(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceTracker:
    """Track performance metrics across multiple stages."""

    def __init__(self, operation_name: str):
        """
        Initialize performance tracker.

        Args:
            operation_name: Name of the overall operation
        """
        self.operation_name = operation_name
        self.stages = {}
        self.start_time = time.time()

    def record_stage(self, stage_name: str, duration: float):
        """
        Record a stage's duration.

        Args:
            stage_name: Name of the stage
            duration: Duration in seconds
        """
        self.stages[stage_name] = duration

    def log_summary(self):
        """Log a summary of all stages."""
        total_time = time.time() - self.start_time

        logger.info("="*60)
        logger.info(f"[PERF] Performance Summary: {self.operation_name}")
        logger.info("="*60)

        for stage_name, duration in self.stages.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            logger.info(
                f"[PERF]   {stage_name:.<40} {duration:>7.3f}s ({percentage:>5.1f}%)"
            )

        logger.info("-"*60)
        logger.info(f"[PERF]   {'TOTAL':.<40} {total_time:>7.3f}s (100.0%)")
        logger.info("="*60)

        return {
            'operation': self.operation_name,
            'total_time': total_time,
            'stages': self.stages
        }

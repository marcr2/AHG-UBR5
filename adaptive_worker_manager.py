#!/usr/bin/env python3
"""
Adaptive Worker Manager for xrvix Processing

This module provides automatic worker scaling based on request rates to optimize
performance while preventing rate limiting issues.
"""

import time
import threading
from collections import deque
from typing import Optional, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveWorkerManager:
    """
    Manages adaptive worker scaling based on request rates.
    
    Starts with 1 worker and automatically scales up/down based on
    the average requests per second over the past minute.
    """
    
    def __init__(self, 
                 target_rps: float = 24,
                 scaling_interval: int = 10,
                 max_workers: int = 20,
                 min_workers: int = 1,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 1.2,
                 stability_periods: int = 3):
        """
        Initialize the adaptive worker manager.
        
        Args:
            target_rps: Target requests per second
            scaling_interval: How often to check for scaling (seconds)
            max_workers: Maximum number of workers allowed
            min_workers: Minimum number of workers
            scale_up_threshold: Scale up if RPS < target * threshold
            scale_down_threshold: Scale down if RPS > target * threshold
            stability_periods: Number of stable periods before scaling up again
        """
        self.target_rps = target_rps
        self.scaling_interval = scaling_interval
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.stability_periods = stability_periods
        
        # Current state
        self.current_workers = self.min_workers
        self.request_times = deque(maxlen=1000)  # Store last 1000 request times
        self.last_scale_time = time.time()
        self.stable_periods = 0
        self.scaling_lock = threading.Lock()
        
        # Scaling history for debugging
        self.scaling_history = []
        
        # Delay before starting adaptive scaling to allow processing to begin
        self.start_time = time.time()
        self.warmup_period = 30  # Wait 30 seconds before starting adaptive scaling
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"AdaptiveWorkerManager initialized: target_rps={target_rps}, "
                   f"scaling_interval={scaling_interval}s, max_workers={max_workers}, "
                   f"warmup_period={self.warmup_period}s")
    
    def record_request(self):
        """Record a request timestamp for rate calculation."""
        with self.scaling_lock:
            self.request_times.append(time.time())
    
    def get_current_rps(self) -> float:
        """Calculate current requests per second over the last 10 seconds (shorter window for faster response)."""
        now = time.time()
        ten_seconds_ago = now - 10  # Use 10-second window instead of 60
        
        with self.scaling_lock:
            # Count requests in the last 10 seconds
            recent_requests = sum(1 for req_time in self.request_times 
                                if req_time >= ten_seconds_ago)
        
        return recent_requests / 10.0  # RPS over 10 seconds
    
    def record_rate_limit_error(self):
        """Record a rate limit error to trigger immediate scale down."""
        with self.scaling_lock:
            # Immediately scale down if we hit rate limits
            if self.current_workers > self.min_workers:
                old_workers = self.current_workers
                self.current_workers = max(self.min_workers, self.current_workers - 1)
                logger.warning(f"🚨 Rate limit hit! Scaled DOWN immediately: {old_workers} → {self.current_workers} workers")
                
                # Record scaling event
                self.scaling_history.append({
                    'timestamp': time.time(),
                    'direction': 'down',
                    'old_workers': old_workers,
                    'new_workers': self.current_workers,
                    'current_rps': self.get_current_rps(),
                    'reason': 'rate_limit_error'
                })
    
    def _should_scale_up(self, current_rps: float) -> bool:
        """Determine if we should scale up workers."""
        threshold = self.target_rps * self.scale_up_threshold
        return (current_rps < threshold and 
                self.current_workers < self.max_workers)
    
    def _should_scale_down(self, current_rps: float) -> bool:
        """Determine if we should scale down workers."""
        threshold = self.target_rps * self.scale_down_threshold
        return (current_rps > threshold and 
                self.current_workers > self.min_workers)
    
    def _scale_workers(self, direction: str):
        """Scale workers up or down."""
        with self.scaling_lock:
            old_workers = self.current_workers
            
            if direction == "up" and self.current_workers < self.max_workers:
                self.current_workers += 1
                logger.info(f"🔼 Scaled UP: {old_workers} → {self.current_workers} workers")
                
            elif direction == "down" and self.current_workers > self.min_workers:
                self.current_workers -= 1
                self.stable_periods = 0  # Reset stability counter only when scaling down
                logger.info(f"🔽 Scaled DOWN: {old_workers} → {self.current_workers} workers")
            
            # Record scaling event
            self.scaling_history.append({
                'timestamp': time.time(),
                'direction': direction,
                'old_workers': old_workers,
                'new_workers': self.current_workers,
                'current_rps': self.get_current_rps()
            })
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            try:
                time.sleep(self.scaling_interval)
                
                if not self.monitoring:
                    break
                
                # Wait for warmup period before starting adaptive scaling
                elapsed = time.time() - self.start_time
                if elapsed < self.warmup_period:
                    logger.debug(f"Warmup period: {elapsed:.1f}s/{self.warmup_period}s")
                    continue
                
                current_rps = self.get_current_rps()
                
                # Check if we should scale
                if self._should_scale_up(current_rps):
                    self._scale_workers("up")
                elif self._should_scale_down(current_rps):
                    self._scale_workers("down")
                else:
                    # We're in a stable period
                    self.stable_periods += 1
                    if self.stable_periods == 1:  # Just became stable
                        logger.info(f"✅ Stable performance: {current_rps:.1f} RPS "
                                  f"with {self.current_workers} workers")
                
                # Log current status
                logger.debug(f"Status: {current_rps:.1f} RPS, {self.current_workers} workers, "
                           f"stable periods: {self.stable_periods}")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_worker_count(self) -> int:
        """Get the current number of workers."""
        return self.current_workers
    
    def get_status(self) -> dict:
        """Get current status information."""
        current_rps = self.get_current_rps()
        return {
            'current_workers': self.current_workers,
            'current_rps': current_rps,
            'target_rps': self.target_rps,
            'stable_periods': self.stable_periods,
            'scaling_history': self.scaling_history[-10:],  # Last 10 scaling events
            'performance_ratio': current_rps / self.target_rps if self.target_rps > 0 else 0
        }
    
    def stop(self):
        """Stop the monitoring thread."""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("AdaptiveWorkerManager stopped")


# Global instance
_adaptive_manager: Optional[AdaptiveWorkerManager] = None


def get_adaptive_manager() -> AdaptiveWorkerManager:
    """Get or create the global adaptive worker manager instance."""
    global _adaptive_manager
    if _adaptive_manager is None:
        from processing_config import (
            TARGET_RPS, SCALING_INTERVAL, MAX_WORKERS_LIMIT, MIN_WORKERS,
            WORKER_SCALE_UP_THRESHOLD, WORKER_SCALE_DOWN_THRESHOLD, STABILITY_PERIODS
        )
        _adaptive_manager = AdaptiveWorkerManager(
            target_rps=TARGET_RPS,
            scaling_interval=SCALING_INTERVAL,
            max_workers=MAX_WORKERS_LIMIT,
            min_workers=MIN_WORKERS,
            scale_up_threshold=WORKER_SCALE_UP_THRESHOLD,
            scale_down_threshold=WORKER_SCALE_DOWN_THRESHOLD,
            stability_periods=STABILITY_PERIODS
        )
    return _adaptive_manager


def record_request():
    """Record a request for rate monitoring."""
    manager = get_adaptive_manager()
    manager.record_request()


def get_worker_count() -> int:
    """Get the current number of workers."""
    manager = get_adaptive_manager()
    return manager.get_worker_count()


def get_status() -> dict:
    """Get current adaptive scaling status."""
    manager = get_adaptive_manager()
    return manager.get_status()


def reset_adaptive_manager():
    """Reset the adaptive worker manager for a new processing run."""
    global _adaptive_manager
    if _adaptive_manager:
        _adaptive_manager.stop()
        _adaptive_manager = None
    # Create a new instance
    return get_adaptive_manager()


def stop_adaptive_manager():
    """Stop the adaptive worker manager."""
    global _adaptive_manager
    if _adaptive_manager:
        _adaptive_manager.stop()
        _adaptive_manager = None 
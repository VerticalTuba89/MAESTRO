#! /usr/bin/env python3
"""
Simulated mower adapter for MAESTRO.
Provides start/stop/status bahavior without hardware.
"""

import time
_state = {"running": False, "last_start": None}

def start(device_id="mower_01", duration_seconds=None):
    """Simulate starting the mower"""
    if _state["running"]:
        return {"ok": False, "msg": "Mower already running."}
    _state["running"] = True
    _state["last_start"] = int(time.time())
    #If duration_seconds provided, we don't actually stop automaticallyin simulation
    return {"ok": True, "msg": f"Mower {device_id} started.", "state": _state.copy()}

def stop(device_id="mower_01"):
    """Simulate stopping the mower."""
    if not _state["running"]:
        return {"ok": False, "msg": "Mower is not running."}
    _state["running"] = False
    return {"ok": True, "msg": f"Mower {device_id} stopped.", "state": _state.copy()}

def status(device_id="mower_01"):
    """Report simulated status."""
    return {"ok": True, "msg": "status", "state": _state.copy()}

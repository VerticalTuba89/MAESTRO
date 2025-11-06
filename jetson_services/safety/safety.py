#! /usr/bin/env python3
"""
Safety Middleware for MAESTRO
Checks if a proposed action is safe to execute based on vision input (people nearby) and basic rule-based policies.
"""

from typing import Dict, Any

def check_action(action: dict, vision: dict):
    """
    action = {"tool": "mower.start", "args": {}}
    vision = {"summary": "...", "detections": [...]}

    Returns:
        {
            "allowed": (bool) True/False,
            "reason": (str) "Explanation string",
            "require_confirmation": (bool) True/False
        }
    """
    tool = action.get("tool", "")

    # Extract labels from YOLO detections
    dets = vision.get("detections", [])
    labels = [d["label"].lower() for d in dets]

    person_present = "person" in labels

    #RULE 1 - Never block "stop" actions
    if "stop" in tool:
        return {
            "allowed": True,
            "require_confirmation": False,
            "reason": "Stopping actions are always safe."
        }

    #RULE 2 - Block machine start if a person is detected
    if "start" in tool:
        if person_present:
            return {
                "allowed": False,
                "require_confirmation": False,
                "reason": "No person detected - safe to execute."
            }
    #Default (if tool not recognized)
    return {
        "allowed": False,
        "require_confirmation": False,
        "reason": f"Unknown tool '{tool}'."
    }

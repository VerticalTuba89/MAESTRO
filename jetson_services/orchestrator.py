#! /usr/bin/env python3
"""
MAESTRO Orchestrator (transcript -> vision -> safety -> adapter)
"""

import os
import sys
import traceback

# --- Add project root to python path ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import local service modules ---
try:
    from jetson_services.audio import audio_service
    from jetson_services.vision import vision_service
    from jetson_services.safety import safety
    from jetson_services.adapters import mower_adapter
except Exception:
    print("ERROR: Failed to import local service modules. Check folder structure.")
    traceback.print_exc()
    sys.exit(1)

def map_transcript_to_action(transcript: str):
    """
    Very small rule-based mapper. Returns None or dict {"tool":..., "args": {...}}
    Extend this with LLM later.
    """
    if not transcript:
        return None
    t= transcript.lower()

    #Mower start/stop moving
    if "start" in t and "mower" in t:
        #Check for optional duration in minutes (EX. "For 10 minutes.")
        import re
        m = re.search(r"for\s+(\d+)\s+minute", t)
        args = {"device_id": "mower_01"}
        if m:
            try:
                args["duration_seconds"] = int(m.group(1)) * 60
            except:
                pass
        return {"tool": "mower.start", "args": args}

    if ("stop" in t or "shutdown" in t or "turn off" in t) and "mower" in t:
        return {"tool": "mower.stop", "args": {"device_id": "mower_01"}}

    #Media example
    if "play" in t and ("song" in t or "music" in t):
        return {"tool": "media.play", "args": {"playlist": "favorites"}}

    #No mapping found
    return None

def execute_action_simulated(action: dict):
    """Call the appropriate adapter (simulation)."""
    tool = action.get("tool", "")
    args = action.get("args", {})

    #Mower adapter
    if tool == "mower.start":
        return mower_adapter.start(**args)
    if tool == "mower.stop":
        return mower_adapter.stop(**args)

    #Unknown tool
    return {"ok": False, "msg": f"No adapter for tool {tool}"}

def main():
    print("MAESTRO orchetsrator: start")

    # 1) Get Transcript
    transcript = None
    try:
        print("\n-- Audio: transcribing")
        transcript = audio_service.transcribe_audio(model_name="base")
        print("Transcript: ", transcript)
    except Exception:
        print("Audio error:")
        traceback.print_exc()

    # 2) Vision snapshot
    vis_res = {"ok": False, "summary": "vision not run", "detections": []}
    try:
        print("\n-- Vision: capture")
        vis_res = vision_service.run_vision_once()
        print("[Vision summary]", vis_res.get("summary"))
    except Exception:
        print("Vision error:")
        traceback.print_exc()

    # 3) Map transcript -> proposed action
    proposed_action = map_transcript_to_action(transcript)
    if not proposed_action:
        print("\nNo actionable command detected in transcript.")
        return

    print("\nProposed action:", proposed_action)

    # 4) Safety check
    result = safety.check_action(proposed_action, vis_res)
    print("Safety result:", result)

    # 5) Decide: blocked / require confirmation / execute
    if not result.get("allowed", False) and not result.get("require_confirmation", False):
        print("ACTION BLOCKED:", result.get("reason"))
        return

    if result.get("require_confirmation", False):
        print("Requires confirmation:", result.get("reason"))
        try:
            confirm = input("Type 'yes' to confirm: ").strip().lower()
        except Exception:
            confirm = ""
        if confirm != "yes":
            print("User canceled.")
            return

    # 6) Execute (simulated)
    exec_res = execute_action_simulated(proposed_action)
    print("Execution result:", exec_res)

    #End
    print("\nMAESTRO orchestrator: Done")

if __name__ == "__main__":
    main()

# rthook_asyncio_winpolicy.py
# Ensure asyncio uses the Selector policy on Windows in frozen apps.
import sys, asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

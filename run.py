"""
run.py — Single command to start the ML Projects server.
Run: python run.py
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 55)
    print("  ML Projects by Abhay Kumar Sharma")
    print("  Starting server on http://localhost:8000")
    print("=" * 55)

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
    )

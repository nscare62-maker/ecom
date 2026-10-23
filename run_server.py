"""
Server runner script for Graph-Based Customer Behavior Modeling.
Supports local execution and cloud deployment platforms (Render, Heroku, Railway).
Usage: python run_server.py [--port 8000] [--host 0.0.0.0]
"""

import os
import argparse
import uvicorn


def main():
    default_host = os.environ.get("HOST", "0.0.0.0" if (os.environ.get("PORT") or os.environ.get("RENDER")) else "127.0.0.1")
    default_port = int(os.environ.get("PORT", 8000))

    parser = argparse.ArgumentParser(description="Run Graph Recommendation Web Server")
    parser.add_argument("--host", type=str, default=default_host, help="Host IP address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"[*] Starting Graph Recommendation Server")
    print(f"[-] Binding: http://{args.host}:{args.port}")
    print(f"=======================================================\n")

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

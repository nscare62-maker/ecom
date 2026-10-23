"""
Server runner script for Graph-Based Customer Behavior Modeling.
Usage: python run_server.py [--port 8000] [--host 127.0.0.1]
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Graph Recommendation Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"[*] Starting Graph Recommendation Server")
    print(f"[-] Web UI: http://{args.host}:{args.port}")
    print(f"[-] OpenAPI Docs: http://{args.host}:{args.port}/docs")
    print(f"=======================================================\n")

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SERVER_APP = REPO_ROOT / "server" / "app" / "main.py"
WEB_DIR = REPO_ROOT / "web"


def stream_output(prefix: str, proc: subprocess.Popen):
	for line in iter(proc.stdout.readline, b""):
		if not line:
			break
		try:
			decoded = line.decode(errors="ignore").rstrip()
			print(f"[{prefix}] {decoded}")
		except Exception:
			pass


def start_server(host: str, port: int, reload: bool) -> subprocess.Popen:
	cmd = [
		sys.executable,
		"-m",
		"uvicorn",
		"server.app.main:app",
		"--host",
		host,
		"--port",
		str(port),
	]
	if reload:
		cmd.append("--reload")
	env = os.environ.copy()
	proc = subprocess.Popen(
		cmd,
		cwd=str(REPO_ROOT),
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		env=env,
	)
	threading.Thread(target=stream_output, args=("server", proc), daemon=True).start()
	return proc


def start_web(port: int) -> subprocess.Popen:
	cmd = ["npm", "run", "dev"]
	env = os.environ.copy()
	# Vite default is 5173; to change, users can edit vite.config or use flags
	proc = subprocess.Popen(
		cmd,
		cwd=str(WEB_DIR),
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		env=env,
	)
	threading.Thread(target=stream_output, args=("web", proc), daemon=True).start()
	return proc


def main():
	parser = argparse.ArgumentParser(description="VoiceLab launcher (server + web)")
	parser.add_argument("--host", default="0.0.0.0", help="Server host")
	parser.add_argument("--server-port", type=int, default=8081, help="FastAPI port")
	parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn reload")
	parser.add_argument("--skip-server", action="store_true", help="Do not start server")
	parser.add_argument("--skip-web", action="store_true", help="Do not start web UI")
	args = parser.parse_args()

	if not SERVER_APP.exists() and not args.skip_server:
		print(f"ERROR: Server app not found at {SERVER_APP}")
		return 1
	if not WEB_DIR.exists() and not args.skip_web:
		print(f"ERROR: Web directory not found at {WEB_DIR}")
		return 1

	procs = []
	try:
		if not args.skip_server:
			print(f"Starting server on {args.host}:{args.server_port}...")
			procs.append(start_server(args.host, args.server_port, reload=(not args.no_reload)))
			time.sleep(0.5)

		if not args.skip_web:
			print("Starting web UI (Vite dev server)...")
			procs.append(start_web(5173))

		print("VoiceLab is running. Press Ctrl+C to stop.")
		while True:
			alive = [p.poll() is None for p in procs]
			if not all(alive):
				break
			time.sleep(0.5)

	finally:
		for p in procs:
			if p.poll() is None:
				try:
					if os.name == "nt":
						p.terminate()
					else:
						p.send_signal(signal.SIGINT)
					time.sleep(0.5)
					if p.poll() is None:
						p.kill()
				except Exception:
					pass

	return 0


if __name__ == "__main__":
	sys.exit(main())


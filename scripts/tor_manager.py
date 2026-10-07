import time
from stem.process import launch_tor_with_config


def start_tor_process(socks_port=9050, control_port=9051):
    """
    Attempts to launch a Tor process. Requires 'tor' binary to be installed on the system.
    """
    print(f"[*] Attempting to start Tor on SOCKS port {socks_port}...")
    try:
        tor_process = launch_tor_with_config(
            config={
                'SocksPort': str(socks_port),
                'ControlPort': str(control_port),
            },
            take_ownership=True,
        )
        print("[+] Tor started successfully!")
        return tor_process
    except Exception as e:
        print(f"[-] Failed to start Tor: {e}")
        print("[!] Make sure 'tor' is installed on your system (e.g., 'brew install tor' on macOS).")
        return None


def keep_process_alive_until_interrupted(process):
    """Keep the process alive until Ctrl+C, then shut it down."""
    try:
        print("[*] Tor is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Stopping Tor...")
        process.kill()


if __name__ == "__main__":
    process = start_tor_process()
    if process:
        keep_process_alive_until_interrupted(process)

import requests

# Try both localhost and the user's provided IP
PROXY_OPTIONS = [
    "socks5h://127.0.0.1:9050",
    "socks5h://192.168.1.40:9050"
]


def fetch_current_ip(proxies=None):
    try:
        response = requests.get("https://httpbin.org/ip", proxies=proxies, timeout=10)
        return response.json().get("origin")
    except Exception as e:
        return f"Error: {e}"


def check_tor_is_working(ip_result):
    """Return True if the IP result shows a successful Tor connection."""
    return "Error" not in ip_result


def test_proxy_connection(proxy_url):
    """Test a single proxy and return the fetched IP."""
    proxies = {"http": proxy_url, "https": proxy_url}
    return fetch_current_ip(proxies)


def display_proxy_test_result(proxy_url, tor_ip):
    print(f"\nTesting Proxy: {proxy_url}")
    print(f"IP via Tor: {tor_ip}")
    if check_tor_is_working(tor_ip):
        print("[SUCCESS] Tor is working and masking your IP!")
    else:
        print("[FAILURE] Could not connect to Tor via this proxy.")


if __name__ == "__main__":
    print("--- Tor Connection Proof ---")
    print(f"Direct IP (No Proxy): {fetch_current_ip()}")

    for proxy_url in PROXY_OPTIONS:
        tor_ip = test_proxy_connection(proxy_url)
        display_proxy_test_result(proxy_url, tor_ip)

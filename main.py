from app.crawler.browser_fetcher import check_tor, renew_tor_circuit
from app.crawler.crawler import crawl
from app.crawler.user_crawler import crawl_user
from app.crawler.crawl_utils import sanitize_filename, filter_substantial_posts, append_to_json_file
from app.parser.date_utils import normalize_timestamps
import json
import os

from app.database import init_db, save_posts


def _persist_crawl_results(data, filename="crawl_data.json"):
    """Append data to a JSON file and save to database."""
    if not data:
        print("No Data Logged!")
        return

    data = normalize_timestamps(data)

    all_data = append_to_json_file(data, filename)
    print(f"Data saved to {filename} (Total records: {len(all_data)})")

    # Sync to database
    save_posts(data)
    print(f"[+] Saved {len(data)} posts to database.")

    return all_data


def run_general_crawl():
    """Mode 1: General forum crawl (original behavior)."""
    url = input("\nEnter URL to crawl: ")

    print("\n=== Crawling ===")
    data = crawl(url, 5)

    all_data = _persist_crawl_results(data)

    if not all_data:
        if os.path.exists("crawl_data.json"):
            with open("crawl_data.json", "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []

    print(f"\nCollected {len(data)} new records")
    return all_data


def _prompt_username_from_url(url_input):
    """Detect a username from a profile URL, prompting for confirmation.

    Returns (forum_url, username) with the resolved values.
    """
    from app.crawler.user_crawler import detect_profile_url
    detected_base, detected_user = detect_profile_url(url_input)

    if detected_user:
        print(f"  [+] Auto-detected username: {detected_user}")
        confirm = input(f"  Use '{detected_user}' as target? (Y/n): ").strip().lower()
        username = detected_user if confirm not in ("n", "no") else input("  Enter target username: ").strip()
        forum_url = url_input  # Pass the full profile URL — crawl_user handles it
    else:
        forum_url = url_input
        username = input("Enter target username: ").strip()

    return forum_url, username


def _prompt_max_pages(default=30):
    """Prompt for a page limit, returning the default on blank or invalid input."""
    raw = input(f"Max pages to crawl (default {default}): ").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def run_user_crawl():
    """Mode 2: User-targeted crawl — collect all posts by a specific user."""
    print("\n  You can provide either:")
    print("    - A direct profile link (e.g., http://tenebris...onion/u/dark00store)")
    print("    - A forum base URL + username separately")

    url_input = input("\nEnter URL: ").strip()
    forum_url, username = _prompt_username_from_url(url_input)
    max_pages = _prompt_max_pages()

    print(f"\n=== Crawling posts by '{username}' ===")
    data = crawl_user(forum_url, username, max_pages)

    if data:
        # Save to user-specific file
        safe_name = sanitize_filename(username)
        user_file = f"user_data_{safe_name}.json"
        _persist_crawl_results(data, user_file)

        # Also save to main crawl_data.json for the ML pipeline
        _persist_crawl_results(data, "crawl_data.json")

        print(f"\nCollected {len(data)} posts by '{username}'")
        print(f"  Saved to: {user_file}")
        print(f"  Also merged into: crawl_data.json")
    else:
        print(f"\nNo posts found for '{username}'")

    return data


def _fetch_db_users():
    """Query all users stored in the database with their platform."""
    try:
        from app.database import DB_PATH
        import sqlite3

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT username, forum, COUNT(*) as post_count
            FROM posts
            WHERE username != 'unknown' AND content IS NOT NULL
            GROUP BY username, forum
            ORDER BY forum, username
        """)
        rows = c.fetchall()
        conn.close()

        return rows
    except Exception as e:
        print(f"\n  [ERROR] Failed to query database: {e}")
        return []


def _print_user_table(rows):
    """Format and display a user table to the console."""
    if not rows:
        print("\n  [!] No users found in the database.")
        print("      Run a crawl first (Mode 1 or 2) to populate the database.")
        return

    print(f"\n  {'-'*60}")
    print(f"  {'#':<5} {'Username':<25} {'Platform / Forum':<25} {'Posts':<6}")
    print(f"  {'-'*60}")
    for i, (username, forum, post_count) in enumerate(rows):
        print(f"  {i+1:<5} {username:<25} {forum:<25} {post_count:<6}")
    print(f"  {'-'*60}")
    print(f"  Total: {len(rows)} user-forum profiles\n")


def _fetch_and_print_users():
    """Fetch users and print them, returning the list if valid."""
    rows = _fetch_db_users()
    _print_user_table(rows)
    if len(rows) < 2:
        print("  [!] Need at least 2 users in the database.")
        return None
    return rows


def _load_or_train_model():
    """Load the trained model, or return None if unavailable."""
    try:
        from Model.model import AuthorSimilarityModel
        from pathlib import Path

        model_path = Path(__file__).resolve().parent / "Model" / "trained_model.joblib"
        if not model_path.exists():
            print("\n  [!] No trained model found at Model/trained_model.joblib")
            print("      Falling back to cosine similarity (less accurate).")
            print("      To train: python -m Model.train --data crawl_data.json\n")
            return None

        return AuthorSimilarityModel.load()
    except Exception as e:
        print(f"\n  [!] Could not load model: {e}")
        print("      Falling back to cosine similarity.\n")
        return None


def _interpret_score(score):
    """Return a human-readable label for a similarity score."""
    if score > 0.7:
        return "LIKELY SAME PERSON", "[+]"
    if score > 0.4:
        return "UNCERTAIN", "[?]"
    return "LIKELY DIFFERENT", "[-]"


def _ensure_fingerprint(uname, uforum):
    """Ensure a fingerprint exists for a given user, building it if missing."""
    from Model.fingerprint_store import get_fingerprint, store_fingerprint
    from app.database import get_user_posts

    if get_fingerprint(uname, uforum) is not None:
        return True

    print(f"  [*] Building fingerprint for {uname}@{uforum}...")
    rows = get_user_posts(uname, uforum)
    posts = filter_substantial_posts(rows)
    if len(posts) < 1:
        print(f"  [ERROR] Not enough substantial posts for {uname}@{uforum} (need 1+).")
        return False
    store_fingerprint(uname, uforum, posts)
    return True


def _compute_similarity_score(fp_a, fp_b, model):
    """Return a float similarity score for two fingerprints."""
    import numpy as np
    diff = np.abs(fp_a["fingerprint"] - fp_b["fingerprint"]).reshape(1, -1)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

    if model and model.is_trained:
        return model.predict_pair(diff)

    from sklearn.metrics.pairwise import cosine_similarity
    return float(cosine_similarity(
        fp_a["fingerprint"].reshape(1, -1),
        fp_b["fingerprint"].reshape(1, -1)
    )[0][0])


def _print_comparison_result(user_a, forum_a, fp_a, user_b, forum_b, fp_b, score):
    """Print a formatted identity-comparison result table."""
    label, icon = _interpret_score(score)
    print(f"\n  {'='*60}")
    print(f"  IDENTITY COMPARISON RESULT")
    print(f"  {'='*60}")
    print(f"  User A:  {user_a} @ {forum_a}  ({fp_a['post_count']} posts)")
    print(f"  User B:  {user_b} @ {forum_b}  ({fp_b['post_count']} posts)")
    print(f"  {'-'*60}")
    print(f"  Similarity Score:  {score:.4f}")
    print(f"  Verdict:           {icon} {label}")
    print(f"  {'='*60}")


def _compare_two_users(user_a, forum_a, user_b, forum_b, model):
    """Compare two specific users using the fingerprint store."""
    try:
        from Model.fingerprint_store import get_fingerprint, init_fingerprint_tables
        init_fingerprint_tables()

        for uname, uforum in [(user_a, forum_a), (user_b, forum_b)]:
            if not _ensure_fingerprint(uname, uforum):
                return

        fp_a = get_fingerprint(user_a, forum_a)
        fp_b = get_fingerprint(user_b, forum_b)

        if fp_a is None or fp_b is None:
            print("  [ERROR] Could not retrieve fingerprints after building. Aborting.")
            return

        score = _compute_similarity_score(fp_a, fp_b, model)
        _print_comparison_result(user_a, forum_a, fp_a, user_b, forum_b, fp_b, score)
    except Exception as e:
        print(f"\n  [ERROR] Comparison failed: {e}")


def _compare_against_all(username, forum, model):
    """Compare a single user against all other users in the database."""
    try:
        from Model.fingerprint_store import find_similar_users, init_fingerprint_tables
        init_fingerprint_tables()

        if not _ensure_fingerprint(username, forum):
            return

        all_db_users = _fetch_db_users()
        if not all_db_users:
            _print_user_table(all_db_users)
            return

        for uname, uforum, _ in all_db_users:
            _ensure_fingerprint(uname, uforum)

        print(f"\n  [*] Comparing {username}@{forum} against all stored profiles...\n")
        matches = find_similar_users(username, forum, model=model, top_n=50)

        if not matches:
            print("  [!] No other users found in the fingerprint store to compare against.")
            return

        print(f"  {'-'*72}")
        print(f"  {'Rank':<6} {'Username':<22} {'Forum':<22} {'Score':<8} {'Verdict':<18}")
        print(f"  {'-'*72}")
        for rank, match in enumerate(matches, 1):
            label, icon = _interpret_score(match["score"])
            print(f"  {rank:<6} {match['username']:<22} {match['forum']:<22} "
                  f"{match['score']:<8.4f} {icon} {label}")
        print(f"  {'-'*72}")
    except Exception as e:
        print(f"\n  [ERROR] Comparison failed: {e}")


def _store_fingerprint_for_crawled_user(username, forum_url, data):
    """Build and store a fingerprint for a freshly crawled user."""
    try:
        from Model.fingerprint_store import store_fingerprint, init_fingerprint_tables
        from urllib.parse import urlparse

        init_fingerprint_tables()

        forum_domain = urlparse(forum_url).netloc if forum_url.startswith("http") else forum_url
        posts = [entry["post"] for entry in data if entry.get("post") and len(entry["post"].split()) >= 5]

        if len(posts) < 1:
            print("  [!] Not enough substantial posts to build a fingerprint.")
            return None

        store_fingerprint(username, forum_domain, posts)
        print(f"  [+] Fingerprint stored for {username}@{forum_domain}")
        return forum_domain
    except Exception as e:
        print(f"  [ERROR] Failed to store fingerprint: {e}")
        return None


def _select_user_from_table(rows, prompt):
    """Prompt the user to pick a row from a displayed table by number."""
    pick = input(prompt).strip()
    idx = int(pick) - 1
    if idx < 0 or idx >= len(rows):
        raise IndexError("Selection out of range.")
    return rows[idx][0], rows[idx][1]


def _crawl_and_compare(model):
    """Crawl a new user from a given URL, store data, then compare."""
    print("\n  -- Step 1: Crawl a New User ------------------------------")
    print("  Provide a forum URL and the target username.")
    print("  You can give a direct profile link or a forum base URL.\n")

    url_input = input("  Enter forum/profile URL: ").strip()
    if not url_input:
        print("  [ERROR] URL cannot be empty.")
        return

    forum_url, username = _prompt_username_from_url(url_input)

    if not username:
        print("  [ERROR] Username cannot be empty.")
        return

    max_pages = _prompt_max_pages()

    print(f"\n  [*] Crawling posts by '{username}' from {forum_url}...")
    try:
        data = crawl_user(forum_url, username, max_pages)
    except Exception as e:
        print(f"  [ERROR] Crawl failed: {e}")
        return

    if not data:
        print(f"  [!] No posts found for '{username}'. Cannot proceed with comparison.")
        return

    safe_name = sanitize_filename(username)
    user_file = f"user_data_{safe_name}.json"
    _persist_crawl_results(data, user_file)
    _persist_crawl_results(data, "crawl_data.json")

    print(f"  [+] Collected {len(data)} posts by '{username}'")

    forum_domain = _store_fingerprint_for_crawled_user(username, forum_url, data)
    if forum_domain is None:
        return

    print("\n  -- Step 2: Compare ------------------------------------------")
    print("  Now choose how to compare the crawled user:\n")
    print("    a. Compare against ONE specific user in the database")
    print("    b. Compare against ALL users in the database")

    compare_choice = input("\n  Select (a/b): ").strip().lower()

    if compare_choice == "a":
        rows = _fetch_and_print_users()
        if not rows:
            return

        try:
            target_user, target_forum = _select_user_from_table(
                rows, "  Enter the number of the user to compare against: "
            )
            _compare_two_users(username, forum_domain, target_user, target_forum, model)
        except Exception as e:
            print(f"  [ERROR] {e}")

    elif compare_choice == "b":
        _compare_against_all(username, forum_domain, model)
    else:
        print("  [!] Invalid choice. Skipping comparison.")


def run_identity_comparison():
    """Mode 3: Full-featured identity comparison with database browsing,
    crawl-then-compare workflow, and one-vs-all matching."""
    print("\n" + "="*60)
    print("  IDENTITY COMPARISON ENGINE")
    print("="*60)
    print()
    print("  a. View users in database & compare two existing users")
    print("  b. Compare one existing user against ALL others")
    print("  c. Crawl a new user first, then compare")
    print("  d. Back to main menu")
    print()

    choice = input("  Select option (a/b/c/d): ").strip().lower()
    if choice == "d":
        return

    model = _load_or_train_model()

    if choice == "a":
        rows = _fetch_and_print_users()
        if not rows:
            return

        try:
            pick1 = input("  Enter number of FIRST user:  ").strip()
            pick2 = input("  Enter number of SECOND user: ").strip()

            idx1 = int(pick1) - 1
            idx2 = int(pick2) - 1

            if idx1 < 0 or idx1 >= len(rows) or idx2 < 0 or idx2 >= len(rows):
                print("  [ERROR] Selection out of range.")
                return

            if idx1 == idx2:
                print("  [!] You selected the same user twice. Choose two different users.")
                return

            user_a, forum_a, _ = rows[idx1]
            user_b, forum_b, _ = rows[idx2]

            _compare_two_users(user_a, forum_a, user_b, forum_b, model)
        except ValueError:
            print("  [ERROR] Please enter valid numbers.")
        except Exception as e:
            print(f"  [ERROR] {e}")

    elif choice == "b":
        rows = _fetch_and_print_users()
        if not rows:
            return

        try:
            username, forum = _select_user_from_table(
                rows, "  Enter the number of the target user: "
            )
            _compare_against_all(username, forum, model)
        except Exception as e:
            print(f"  [ERROR] {e}")

    elif choice == "c":
        _crawl_and_compare(model)
    else:
        print("  [!] Invalid option. Returning to main menu.")


def _print_main_menu():
    """Print the top-level application menu with banner."""
    import os
    if os.name == 'nt':
        os.system("") # Enable VT100 ANSI sequences on Windows
        
    CYAN = '\033[1;36m'
    MAGENTA = '\033[1;35m'
    RESET = '\033[0m'
    
    banner = f"""
{CYAN} ███╗   ███╗███╗   ███╗███████╗██╗ ██████╗ 
 ████╗ ████║████╗ ████║██╔════╝██║██╔════╝ 
 ██╔████╔██║██╔████╔██║█████╗  ██║██║      
 ██║╚██╔╝██║██║╚██╔╝██║██╔══╝  ██║██║      
 ██║ ╚═╝ ██║██║ ╚═╝ ██║██║     ██║╚██████╗ 
 ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ {RESET}
{MAGENTA}           :: T R O W L E R ::{RESET}
{CYAN}    by D-Fault (https://github.com/TheFault666){RESET}
    """
    print(banner)
    print("="*60)
    print("  1. General Forum Crawl")
    print("     Crawl all threads from a forum URL")
    print()
    print("  2. User-Targeted Crawl")
    print("     Collect ALL posts by a specific username")
    print()
    print("  3. Identity Comparison Engine")
    print("     Compare users, crawl & compare, or match against all")
    print()


def main_menu():
    # Initialize database
    init_db()

    print("=== Checking Tor ===")
    if check_tor():
        print("\n=== Getting fresh Tor identity ===")
        renew_tor_circuit()

    _print_main_menu()
    choice = input("Select mode (1/2/3): ").strip()

    if choice == "1":
        run_general_crawl()
    elif choice == "2":
        run_user_crawl()
    elif choice == "3":
        run_identity_comparison()
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        from app.crawler.browser_fetcher import close_browser
        close_browser()
        input("\nPress Enter to exit...")

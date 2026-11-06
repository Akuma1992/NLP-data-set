"""Log in to Hugging Face with the token on the clipboard, without pasting into the terminal.

Pasting into some terminals adds hidden control characters, which makes Hugging Face reject
the token ("Invalid user token"). This reads the clipboard only after you press Enter,
picks out the hf_... token, saves the login, and prints the account name - never the token.
"""
import re
import subprocess
import sys

from huggingface_hub import login, whoami

input("Click Copy on the token popup in the browser, then press Enter here... ")
raw = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
                     capture_output=True, text=True).stdout or ""
m = re.search(r"hf_[A-Za-z0-9]{20,}", raw)
if not m:
    print(f"No Hugging Face token (hf_...) found on the clipboard "
          f"({len(raw.strip())} characters there). Copy it again and re-run this.")
    sys.exit(1)
try:
    login(token=m.group(0), add_to_git_credential=False)
except Exception as e:  # message from Hugging Face, contains no token
    print("Hugging Face refused this token:", str(e).splitlines()[-1])
    sys.exit(1)
w = whoami()
print(f"Login successful as {w['name']} (token type: {w['auth']['accessToken'].get('role')})")

"""Kuyrukta fotoğraf var mı? (Kurulum yapmadan, birkaç saniyede bakar.)
Boşsa iş akışı ağır kurulumu hiç yapmadan biter; ücretsiz dakikalar harcanmaz."""
import json, os, sys, urllib.request

url, key = os.environ.get("APPS_SCRIPT_URL", ""), os.environ.get("FOTO_KEY", "")
out = os.environ.get("GITHUB_OUTPUT")
n = 0
try:
    req = urllib.request.Request(url, data=json.dumps({"action": "fotoQueue", "key": key, "limit": 1}).encode(),
                                 headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(req, timeout=60) as r:
        j = json.loads(r.read().decode("utf-8"))
    if not j.get("ok"):
        print("Apps Script hatası:", j.get("message")); sys.exit(1)
    n = int(j.get("total", 0))
except SystemExit:
    raise
except Exception as e:
    print("Apps Script'e ulaşılamadı:", e); sys.exit(1)
print("Bekleyen fotoğraf:", n)
if out:
    with open(out, "a") as f:
        f.write(f"bekleyen={n}\n")

"""
Sihir Dükkanı — OTOMATİK ÜRÜN FOTOĞRAFI DÜZELTME
GitHub Actions'ta her 30 dakikada bir çalışır (bilgisayarınızın açık olması gerekmez).

Her bekleyen fotoğraf için:
  1) Orijinal fotoğraf Apps Script'ten alınır.
  2) Arka plan yapay zekâyla silinir (rembg / BiRefNet — ücretsiz, açık kaynak, MIT lisansı).
  3) Cloudflare yapay zekâsı (FLUX.2, günde ~90 ücretsiz) fotoğrafı stüdyo fotoğrafı gibi yeniden düzenler.
     YAZI KORUMA: yapay zekânın verdiği duruş/ışık alınır, ürünün kendisi (yazı, logo, etiket) orijinalden gelir.
     Yapay zekâ ürünün şeklini değiştirirse onun sonucu kullanılmaz, kendi düzeltmemiz kullanılır.
  4) Yamuksa düzeltilir, ışık/renk dengelenir, beyaz zemine ortalanır, hafif gölge eklenir (1200×1200 JPG).
  5) Sonuç Drive'a kaydedilir; orijinal "Orijinaller" klasörüne taşınır.

Gizli anahtarlar (GitHub → Settings → Secrets and variables → Actions):
  APPS_SCRIPT_URL  : fiyat sorgu uygulamasının Apps Script adresi (.../exec)
  FOTO_KEY         : E-Tablo menüsündeki "Otomatik fotoğraf anahtarı"
  CF_URL, CF_SIFRE : Cloudflare Worker adresi ve şifresi (yoksa yapay zekâ adımı atlanır)
"""
import base64, io, json, os, sys, time, traceback

import numpy as np
import cv2
import requests
from PIL import Image, ImageOps

APPS_SCRIPT_URL = os.environ.get("APPS_SCRIPT_URL", "").strip()
FOTO_KEY = os.environ.get("FOTO_KEY", "").strip()
CF_URL = os.environ.get("CF_URL", "").strip()
CF_SIFRE = os.environ.get("CF_SIFRE", "").strip()
CF_MODEL = os.environ.get("CF_MODEL", "").strip() or "@cf/black-forest-labs/flux-2-klein-4b"
MAX_PER_RUN = int(os.environ.get("MAX_PER_RUN", "") or 25)
TIME_BUDGET = int(os.environ.get("TIME_BUDGET_SN", "") or 25 * 60)      # bir turda en fazla 25 dk
REMBG_MODEL = os.environ.get("REMBG_MODEL", "").strip() or "birefnet-general-lite"
OUT = 1200                # çıktı: 1200×1200 kare
PAD = 0.06                # kenar boşluğu (her yandan)
MIN_IOU = 0.85            # yapay zekâ sonucu orijinale bu kadar benzemiyorsa kullanılmaz

PROMPT = (
    "Turn this into a professional e-commerce studio product photo of this exact same product. "
    "Keep the product identical: same shape, proportions, colors, printed text, logos, labels and every detail. "
    "Straighten the product so it stands upright and faces the camera, center it, "
    "pure seamless white background (#FFFFFF), soft even studio lighting, natural colors, sharp focus. "
    "Remove hands, hangers, clips, price tags and any other objects. Do not add anything."
)


class QuotaError(Exception):
    pass


# ---------------------------------------------------------------- Apps Script
def api(action, timeout=120, **kw):
    body = dict(kw, action=action, key=FOTO_KEY)
    last = None
    for attempt in range(3):
        try:
            r = requests.post(APPS_SCRIPT_URL, data=json.dumps(body), headers={"Content-Type": "text/plain"}, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # geçici ağ hatası: tekrar dene
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Apps Script'e ulaşılamadı: {last}")


# ---------------------------------------------------------------- arka plan silme
_session = None


def mask_of(rgb):
    """Ürün maskesi (0–255). rembg / BiRefNet, işlemcide çalışır."""
    global _session
    from rembg import new_session, remove
    if _session is None:
        _session = new_session(REMBG_MODEL)
    m = remove(Image.fromarray(rgb), session=_session, only_mask=True)
    m = np.array(m.convert("L"))
    if m.shape[:2] != rgb.shape[:2]:
        m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return m


def clean_mask(m):
    """Kenarı sertleştir, kopuk küçük lekeleri sil."""
    a = np.clip((m.astype(np.float32) - 16) / (240 - 16), 0, 1)
    a = a * a * (3 - 2 * a)
    A = (a * 255).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((A > 128).astype(np.uint8), 8)
    if n <= 1:
        return A, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= max(50, areas.max() * 0.03)
    keep_mask = keep[lab]
    keep_mask = cv2.dilate(keep_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    A[~keep_mask] = 0
    A[A < 12] = 0
    return A, int(keep.sum())


def bbox(A, thr=40):
    ys, xs = np.where(A > thr)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


# ---------------------------------------------------------------- renk / ışık
def bg_mean(rgb, A):
    sel = A == 0
    if sel.sum() < A.size * 0.01:
        return None
    return rgb[sel].reshape(-1, 3).mean(0)


def decontaminate(rgb, A, bg):
    """Kenarlarda kalan eski zemin rengini temizle."""
    if bg is None:
        return rgb
    a = A.astype(np.float32) / 255.0
    sel = (A >= 26) & (A <= 249)
    out = rgb.astype(np.float32)
    t = a[sel][:, None]
    out[sel] = np.clip((out[sel] - (1 - t) * bg[None, :]) / t, 0, 255)
    return out.astype(np.uint8)


def color_fix(rgb, A, bg, vivid=0.15):
    """Hafif beyaz dengesi + seviyeler + canlılık. Ürünün gerçek rengini bozmayacak kadar temkinli."""
    f = rgb.astype(np.float32)
    gains = np.ones(3, np.float32)
    if bg is not None:
        L, chroma = bg.mean(), bg.max() - bg.min()
        if L > 60 and 4 < chroma < 32:
            gains = np.clip(1 + 0.6 * (L / np.maximum(bg, 1) - 1), 0.9, 1.12).astype(np.float32)
    f = f * gains
    prod = A > 160
    if prod.sum() > 500:
        lum = (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[prod]
        lo, hi, mean = np.percentile(lum, 0.4), np.percentile(lum, 99.6), float(lum.mean())
        sc, lift = 1.0, 0.0
        if hi - lo > 20:
            sc = float(np.clip(235 / max(1, hi - lo), 1, 1.22))
            if mean < 105:
                lift = min(18.0, (105 - mean) * 0.35)
            top = mean + lift + (hi - mean) * sc
            if top > 250:
                sc = max(1.0, (250 - mean - lift) / max(1, hi - mean))
        f = mean + lift + (f - mean) * sc
    if vivid > 0:
        f = np.clip(f, 0, 255)
        mx, mn, m = f.max(-1, keepdims=True), f.min(-1, keepdims=True), f.mean(-1, keepdims=True)
        sat = (mx - mn) / 255
        k = 1 + vivid * (1 - sat) * (1 - np.abs(m - 128) / 160)
        f = m + (f - m) * k
    out = rgb.copy()
    sel = A > 0
    out[sel] = np.clip(f[sel], 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- yamuk düzeltme
def rotate(img, deg, border=0, interp=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    W, H = int(round(w * c + h * s)), int(round(w * s + h * c))
    M[0, 2] += W / 2 - w / 2
    M[1, 2] += H / 2 - h / 2
    bv = border if np.ndim(border) == 0 else tuple(int(x) for x in border)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)


def tilt_angle(A):
    """Ürünün en dar dikdörtgenine göre eğim (derece, cv2 yönünde). 0 = düzeltme gerekmez. Yuvarlak ürünlere dokunmaz."""
    pts = cv2.findNonZero((A > 128).astype(np.uint8))
    if pts is None or len(pts) < 50:
        return 0.0
    hull = cv2.convexHull(pts)
    (_, _), (rw, rh), ang = cv2.minAreaRect(hull)
    x, y, w, h = cv2.boundingRect(hull)
    if rw * rh >= w * h * 0.965:
        return 0.0
    ang = ((ang + 45) % 90) - 45                     # [-45, 45)
    if not (1.2 <= abs(ang) <= 25):
        return 0.0
    # yönü deneyerek seç: düzeltilmiş halin sınır kutusu küçülmeli
    best, best_area = 0.0, w * h
    for d in (ang, -ang):
        r = rotate(A, d)
        b = bbox(r, 128)
        if b and b[2] * b[3] < best_area:
            best, best_area = d, b[2] * b[3]
    return best


# ---------------------------------------------------------------- Cloudflare yapay zekâsı
def cf_edit(rgb):
    h, w = rgb.shape[:2]
    k = min(1.0, 508 / max(w, h))                     # model 512 px'ten küçük giriş istiyor
    small = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, "JPEG", quality=95)
    r_ = w / h
    ow, oh = 1024, 1024
    if r_ > 1:
        oh = round(1024 / r_ / 16) * 16
    else:
        ow = round(1024 * r_ / 16) * 16
    ow, oh = int(np.clip(ow, 256, 1024)), int(np.clip(oh, 256, 1024))
    try:
        r = requests.post(CF_URL, headers={"X-Sifre": CF_SIFRE},
                          files={"image": ("foto.jpg", buf.getvalue(), "image/jpeg")},
                          data={"prompt": PROMPT, "model": CF_MODEL, "width": str(ow), "height": str(oh)}, timeout=180)
    except Exception as e:
        raise RuntimeError(f"Cloudflare'e ulaşılamadı: {e}")
    try:
        j = r.json()
    except Exception:
        j = {}
    if r.status_code == 429 or j.get("limit"):
        raise QuotaError(j.get("hata") or "Günlük ücretsiz yapay zekâ hakkı doldu")
    if r.status_code != 200 or not j.get("image"):
        raise RuntimeError("Cloudflare: " + str(j.get("hata") or f"HTTP {r.status_code}"))
    data = j["image"]
    if "," in data[:80]:
        data = data.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    return np.array(img)


# ---------------------------------------------------------------- yazı koruma (hizalama + birleştirme)
def _low(A, mx=160):
    h, w = A.shape
    k = min(1.0, mx / max(w, h))
    L = cv2.resize((A > 128).astype(np.uint8), (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_NEAREST)
    return L, k


def _moments(A):
    ys, xs = np.where(A > 128)
    if not len(xs):
        return None
    return len(xs), xs.mean(), ys.mean()


def _affine(deg, s, co, ca):
    """p = ca + s·R(deg)·(q − co)  (görüntü koordinatlarında, y aşağı)"""
    t = np.deg2rad(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) * s
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R
    M[:, 2] = np.array(ca) - R @ np.array(co)
    return M


def align_masks(Ao, Aa):
    """Orijinal maskeyi yapay zekâ maskesine oturtan açı ve ölçek. Dönüş: (iou, deg, s, co, ca) — tam çözünürlükte."""
    mo, ma = _moments(Ao), _moments(Aa)
    if not mo or not ma:
        return None
    Lo, ko = _low(Ao)
    La, ka = _low(Aa)
    s0 = np.sqrt(ma[0] / mo[0])
    co_l, ca_l = (mo[1] * ko, mo[2] * ko), (ma[1] * ka, ma[2] * ka)
    Hh, Ww = La.shape
    la = La.astype(bool)
    best = (-1.0, 0.0, s0)

    def test(d, sm):
        nonlocal best
        s = s0 * sm
        M = _affine(d, s * ka / ko, co_l, ca_l)
        w_ = cv2.warpAffine(Lo, M, (Ww, Hh), flags=cv2.INTER_NEAREST).astype(bool)
        inter, uni = np.logical_and(w_, la).sum(), np.logical_or(w_, la).sum()
        iou = inter / max(1, uni)
        if iou > best[0]:
            best = (iou, d, s)

    for d in range(-25, 26):
        for sm in (0.92, 0.96, 1.0, 1.04, 1.08):
            test(float(d), sm)
    d0, s1 = best[1], best[2] / s0
    for d in np.arange(d0 - 0.75, d0 + 0.76, 0.25):
        for sm in (s1 * 0.98, s1, s1 * 1.02):
            test(float(d), sm)
    return best[0], best[1], best[2], (mo[1], mo[2]), (ma[1], ma[2])


def match_tone(Wp, Wa, Ap, Aa, strength=0.85):
    """Orijinalin rengini/ışığını yapay zekâ sonucuna benzet (kanal başına ortalama + kontrast)."""
    sel = (Wa > 200) & (Aa > 200)
    if sel.sum() < 200:
        return Wp
    w, a = Wp[sel].astype(np.float32), Ap[sel].astype(np.float32)
    mw, ma_, dw, da = w.mean(0), a.mean(0), w.std(0) + 1, a.std(0) + 1
    g = np.clip(da / dw, 0.75, 1.35)
    f = Wp.astype(np.float32)
    t = (f - mw) * g + ma_
    f = f + (t - f) * strength
    return np.clip(f, 0, 255).astype(np.uint8)


def merge_keep_text(rgb, Ao, ai_rgb, Aa):
    """Duruş + temiz sınır + ışık yapay zekâdan, ürünün pikselleri orijinalden."""
    r = align_masks(Ao, Aa)
    if not r:
        return None, 0.0
    iou, deg, s, co, ca = r
    if iou < MIN_IOU:
        return None, iou
    f = float(np.clip(1 / s, 1, 3))                    # yazılar küçülüp bulanıklaşmasın: sonucu büyüt
    H, W = int(round(ai_rgb.shape[0] * f)), int(round(ai_rgb.shape[1] * f))
    big_rgb = cv2.resize(ai_rgb, (W, H), interpolation=cv2.INTER_LANCZOS4)
    big_A = cv2.resize(Aa, (W, H), interpolation=cv2.INTER_LINEAR)
    M = _affine(deg, s * f, co, (ca[0] * f, ca[1] * f))
    w_rgb = cv2.warpAffine(rgb, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    w_A = cv2.warpAffine(Ao, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    w_rgb = match_tone(w_rgb, w_A, big_rgb, big_A, 0.85)
    a = (w_A.astype(np.float32) / 255)[..., None]
    out = (w_rgb * a + big_rgb * (1 - a)).astype(np.uint8)   # saydamlık yapay zekâdan: askı vb. temiz kalır
    return (out, big_A), iou


# ---------------------------------------------------------------- beyaz zemine yerleştirme
def compose(rgb, A):
    b = bbox(A)
    if not b:
        raise RuntimeError("Ürün bulunamadı")
    x, y, w, h = b
    crop, ca = rgb[y:y + h, x:x + w], A[y:y + h, x:x + w]
    shadow = 0.02
    s = min(OUT * (1 - 2 * PAD) / w, OUT * (1 - 2 * PAD - shadow) / h)
    dw, dh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (dw, dh), interpolation=interp)
    ca = cv2.resize(ca, (dw, dh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((OUT, OUT, 3), 255, np.float32)
    dx, dy = (OUT - dw) // 2, (OUT - dh) // 2
    # hafif yumuşak gölge
    sh = np.zeros((OUT, OUT), np.float32)
    rx, ry = dw * 0.46, max(4, min(dw * 0.05, OUT * 0.03))
    cv2.ellipse(sh, (OUT // 2, int(dy + dh - ry * 0.25)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(2, ry * 0.8)) * 0.2
    canvas *= (1 - sh[..., None])
    a = (ca.astype(np.float32) / 255)[..., None]
    region = canvas[dy:dy + dh, dx:dx + dw]
    canvas[dy:dy + dh, dx:dx + dw] = crop * a + region * (1 - a)
    buf = io.BytesIO()
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=90, optimize=True, progressive=True)
    return buf.getvalue()


def qc(A):
    cov = float((A > 128).mean())
    if cov < 0.015:
        return "Ürün bulunamadı ya da çok küçük"
    if cov > 0.92:
        return "Arka plan ayrılamadı"
    return ""


# ---------------------------------------------------------------- tek fotoğraf
def load(data_b64):
    img = Image.open(io.BytesIO(base64.b64decode(data_b64)))
    img = ImageOps.exif_transpose(img).convert("RGB")
    k = min(1.0, 2000 / max(img.size))
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    return np.array(img)


def process(rgb, use_ai, last_try):
    notes = []
    m = mask_of(rgb)
    A, _parts = clean_mask(m)
    err = qc(A)
    if err:
        raise RuntimeError(err)
    bg = bg_mean(rgb, A)
    rgb = decontaminate(rgb, A, bg)

    if use_ai:
        try:
            ai = cf_edit(rgb)
            Aa, _ = clean_mask(mask_of(ai))
            if not qc(Aa):
                res, iou = merge_keep_text(rgb, A, ai, Aa)
                if res is not None:
                    notes.append(f"Yapay zekâ + yazılar korundu (uyum %{round(iou * 100)})")
                    return compose(*res), notes
                notes.append(f"Yapay zekâ ürünün şeklini değiştirdi (uyum %{round(iou * 100)}), kendi düzeltmemiz kullanıldı")
            else:
                notes.append("Yapay zekâ sonucu ayrılamadı, kendi düzeltmemiz kullanıldı")
        except QuotaError:
            raise
        except Exception as e:
            if not last_try:
                raise
            notes.append(f"Yapay zekâ çalışmadı ({str(e)[:80]}), kendi düzeltmemiz kullanıldı")

    # yapay zekâsız yol: düzelt + renk + beyaz zemin
    rgb = color_fix(rgb, A, bg)
    deg = tilt_angle(A)
    if deg:
        rgb = rotate(rgb, deg, border=(255, 255, 255), interp=cv2.INTER_CUBIC)
        A = rotate(A, deg)
        notes.append(f"Düzeltildi {abs(deg):.1f}°")
    return compose(rgb, A), notes


# ---------------------------------------------------------------- ana döngü
def main():
    if not APPS_SCRIPT_URL or not FOTO_KEY:
        print("APPS_SCRIPT_URL ve FOTO_KEY gizli anahtarları girilmemiş.")
        sys.exit(1)
    use_ai = bool(CF_URL and CF_SIFRE)
    print("Yapay zekâ (Cloudflare):", "açık" if use_ai else "kapalı (CF_URL / CF_SIFRE yok)")
    t0 = time.time()
    q = api("fotoQueue", limit=MAX_PER_RUN)
    if not q.get("ok"):
        print("Kuyruk alınamadı:", q.get("message"))
        sys.exit(1)
    items = q.get("items", [])
    print(f"Bekleyen: {q.get('total', 0)} · bu turda: {len(items)}")
    done = fail = 0
    for it in items:
        if time.time() - t0 > TIME_BUDGET:
            print("Süre doldu, kalanlar bir sonraki turda.")
            break
        name = f"{it.get('code') or it['id']} ({it.get('name', '')[:40]})"
        try:
            g = api("fotoGet", id=it["id"])
            if not g.get("ok"):
                raise RuntimeError(g.get("message") or "Fotoğraf alınamadı")
            rgb = load(g["data"])
            t1 = time.time()
            jpg, notes = process(rgb, use_ai, last_try=int(it.get("tries", 0)) >= 2)
            p = api("fotoPut", id=it["id"], data=base64.b64encode(jpg).decode(), note=" · ".join(notes))
            if not p.get("ok"):
                raise RuntimeError(p.get("message") or "Kaydedilemedi")
            done += 1
            print(f"✓ {name} — {time.time() - t1:.1f} sn — {' · '.join(notes) or 'tamam'}")
        except QuotaError as e:
            print(f"⏸ {e} — kalan fotoğraflar hak yenilenince işlenecek.")
            break
        except Exception as e:
            fail += 1
            print(f"✗ {name}: {e}")
            traceback.print_exc()
            try:
                api("fotoFail", id=it["id"], message=str(e)[:200])
            except Exception:
                pass
    print(f"Bitti: {done} işlendi, {fail} hata, {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()

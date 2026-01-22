import os
import time
import uuid
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import streamlit as st

# Optional imports (keep app runnable even if some libs missing)
try:
    import numpy as np
    import cv2
    from PIL import Image
except Exception:
    np = None
    cv2 = None
    Image = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_image_comparison import image_comparison
except Exception:
    image_comparison = None


# =========================================================
# 0) Basic setup / Branding
# =========================================================
APP_TITLE = "AI 術後追蹤系統"
CLINIC_PIN = os.environ.get("AIMED_CLINIC_PIN", "1234")  # set env for production

# IMPORTANT: keep backward compatibility with your previous deployments
# - app1.py used beauty_tracker.db
# - app.py used medical.db
# This auto-select avoids "資料跑掉" when you renamed the file.
DB_PATH = "beauty_tracker.db" if os.path.exists("beauty_tracker.db") else "medical.db"

DATA_DIR = "user_data"
UPLOAD_DIR = "uploads"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title=APP_TITLE, layout="wide")

# =========================================================
# Session defaults (確保客戶端/診所端分流在首次進站就生效)
# =========================================================
st.session_state.setdefault("role", "client")        # 'client' or 'clinic'
st.session_state.setdefault("clinic_authed", False)
st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("user_id", None)
st.session_state.setdefault("alert_confirm_open", False)


st.markdown(
    """

<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }
.stApp { background:#f6f7fb; }

.card, .photo-card, .panel-card{
    background:white;
    border-radius:20px;
    padding:18px;
    box-shadow:0 8px 24px rgba(0,0,0,0.06);
}
.panel-title{font-weight:700;font-size:16px;margin-bottom:10px;}
.metric-card{
    background:linear-gradient(135deg,#667eea,#764ba2);
    color:white;border-radius:16px;padding:18px;
}
</style>

""",
    unsafe_allow_html=True,
)


# =========================================================
# 1) DB + migration (auto add missing columns)
# =========================================================
def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def table_columns(conn, table: str) -> set:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    names = set()
    for r in rows:
        try:
            names.add(r["name"])
        except Exception:
            names.add(r[1])
    return names


def ensure_columns(conn, table: str, columns_sql: dict):
    existing = table_columns(conn, table)
    cur = conn.cursor()
    for col, col_type in columns_sql.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    conn.commit()


def db_init_and_migrate():
    conn = db_conn()
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            phone TEXT,
            name TEXT,
            treatment TEXT,
            op_date TEXT,
            created_at TEXT,
            before_img_path TEXT
        )
        """
    )

    # Records: store both postop_date (computed) and uploaded_at (actual save time)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            record_date TEXT,
            postop_date TEXT,
            uploaded_at TEXT,
            img_path TEXT,
            q_score INTEGER,
            confidence INTEGER,
            wrinkle INTEGER,
            spot INTEGER,
            redness INTEGER,
            pore INTEGER,
            texture INTEGER,
            note TEXT,
            UNIQUE(user_id, stage),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    # Appointments
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            appt_dt TEXT NOT NULL,
            note TEXT,
            status TEXT,
            created_at TEXT,
            UNIQUE(user_id, appt_dt),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    # Alerts: add cancellation + contact preference fields
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            stage TEXT,
            severity TEXT,
            reason TEXT,
            symptoms TEXT,
            metrics_json TEXT,
            img_path TEXT,
            resolved INTEGER DEFAULT 0,
            status TEXT,
            updated_at TEXT,
            canceled_at TEXT,
            cancel_reason TEXT,
            contact_method TEXT,
            contact_time TEXT,
            no_call INTEGER,
            user_note TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    conn.commit()

    ensure_columns(conn, "users", {
        "phone": "TEXT",
        "name": "TEXT",
        "treatment": "TEXT",
        "op_date": "TEXT",
        "created_at": "TEXT",
        "before_img_path": "TEXT",
    })

    ensure_columns(conn, "records", {
        "record_date": "TEXT",
        "postop_date": "TEXT",
        "uploaded_at": "TEXT",
        "img_path": "TEXT",
        "q_score": "INTEGER",
        "confidence": "INTEGER",
        "wrinkle": "INTEGER",
        "spot": "INTEGER",
        "redness": "INTEGER",
        "pore": "INTEGER",
        "texture": "INTEGER",
        "note": "TEXT",
    })

    ensure_columns(conn, "appointments", {
        "note": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
    })

    ensure_columns(conn, "alerts", {
        "stage": "TEXT",
        "severity": "TEXT",
        "reason": "TEXT",
        "symptoms": "TEXT",
        "metrics_json": "TEXT",
        "img_path": "TEXT",
        "resolved": "INTEGER",
        "status": "TEXT",
        "updated_at": "TEXT",
        "canceled_at": "TEXT",
        "cancel_reason": "TEXT",
        "contact_method": "TEXT",
        "contact_time": "TEXT",
        "no_call": "INTEGER",
        "user_note": "TEXT",
    })

    conn.close()


db_init_and_migrate()


# =========================================================
# 1.1) Fusion: 855版「客戶/療程(episode)」資料模型（不破壞既有 users/records）
#  - 目的：保留既有功能的完整度，同時提供更像真實醫美網站的客戶資料結構
#  - 策略：clients/episodes 作為前台入口；選定 episode 後，同步/對應到既有 users 表，讓原本功能全部可用
# =========================================================
def init_clients_episodes_tables():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS clients (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT,
        created_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        client_id TEXT NOT NULL,
        procedure_json TEXT NOT NULL,
        surgery_date TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(client_id) REFERENCES clients(id)
    )
    """)

    # Optional: store photo blob (for quick preview) but keep legacy file-path flow for analytics/history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS episode_photos (
        id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL,
        kind TEXT NOT NULL, -- 'before' or 'followup'
        taken_date TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        image_png BLOB,
        img_path TEXT,
        meta_json TEXT,
        FOREIGN KEY(episode_id) REFERENCES episodes(id)
    )
    """)

    conn.commit()
    conn.close()

def _now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def list_clients_v3(limit=200):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM clients ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_client_v3(name: str, phone: str = ""):
    cid = uuid.uuid4().hex
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO clients (id,name,phone,created_at) VALUES (?,?,?,?)",
        (cid, name.strip(), (phone.strip() or None), _now_iso()),
    )
    conn.commit()
    conn.close()
    return cid

def list_episodes_v3(client_id: str, limit=200):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM episodes WHERE client_id=? ORDER BY surgery_date DESC, created_at DESC LIMIT ?",
        (client_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_episode_v3(client_id: str, procedures: list, surgery_date: str):
    eid = uuid.uuid4().hex
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO episodes (id,client_id,procedure_json,surgery_date,created_at) VALUES (?,?,?,?,?)",
        (eid, client_id, json.dumps(procedures, ensure_ascii=False), surgery_date, _now_iso()),
    )
    conn.commit()
    conn.close()
    return eid

def get_client_v3(cid: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM clients WHERE id=?", (cid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_episode_v3(eid: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM episodes WHERE id=?", (eid,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def ensure_user_from_episode(eid: str):
    """
    將 episode（client/療程）同步到既有 users 表，讓 1600 行版的 tracking/report/alerts 全部可用。
    user_id 採用 episode_id（穩定、可重複登入）。
    """
    ep = get_episode_v3(eid)
    if not ep:
        return None
    c = get_client_v3(ep["client_id"])
    if not c:
        return None

    user_id = eid  # map: 1 episode = 1 user context
    name = c.get("name") or ""
    phone = c.get("phone") or ""
    treatment = " + ".join(json.loads(ep["procedure_json"])) if ep.get("procedure_json") else ""
    op_date = ep.get("surgery_date") or date.today().isoformat()

    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE users SET phone=?, name=?, treatment=?, op_date=? WHERE user_id=?",
            (phone, name, treatment, op_date, user_id),
        )
    else:
        cur.execute(
            """
            INSERT INTO users (user_id, phone, name, treatment, op_date, created_at, before_img_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, phone, name, treatment, op_date, _now_str(), None),
        )
    conn.commit()
    conn.close()
    return user_id

init_clients_episodes_tables()


# =========================================================
# 2) Data model helpers
# =========================================================
STAGES = [
    "術後第 1 天",
    "術後第 2 天",
    "術後第 3 天",
    "術後第 7 天",
    "術後第 14 天",
    "術後第 30 天",
    "術後 30 天以上",
]


def stage_order(stage: str) -> int:
    if stage in STAGES:
        return STAGES.index(stage)
    return 999


def stage_to_days(stage: str):
    """Extract N from '術後第 N 天'. Return None for non-fixed stages (e.g., '術後 30 天以上')."""
    if not stage:
        return None
    m = re.search(r"第\s*(\d+)\s*天", stage)
    if m:
        return int(m.group(1))
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def metrics_avg(m: dict) -> int:
    if not m:
        return 0
    return int(round(sum(m.values()) / len(m)))


def improvement_pct(curr: int, base: int) -> int:
    if base is None or base <= 0:
        return 0
    return int(round(((curr - base) / base) * 100))


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================================================
# 3) Image helpers
# =========================================================
def save_rgb_image(rgb_np, prefix="img") -> str:
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(DATA_DIR, fname)
    if Image is None:
        return path
    Image.fromarray(rgb_np).save(path, quality=95)
    return path


def save_uploaded_image(file, prefix="upload") -> str:
    if Image is None:
        return ""
    img = Image.open(file).convert("RGB")
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(UPLOAD_DIR, fname)
    img.save(path)
    return path


def load_image_rgb(file_or_path):
    if file_or_path is None:
        return None
    if np is None or Image is None:
        return None
    if isinstance(file_or_path, str):
        if not os.path.exists(file_or_path):
            return None
        img = Image.open(file_or_path).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")
    return np.array(img)


def calc_quality_score_simple(img_pil) -> int:
    if np is None or img_pil is None:
        return 70
    arr = np.array(img_pil.convert("RGB")).astype("float32")
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
    mean = float(lum.mean())
    std = float(lum.std())
    score = 100
    if mean < 60 or mean > 200:
        score -= 25
    if std < 25:
        score -= 20
    return int(clamp(score, 40, 100))


def mock_metrics(img_pil):
    if np is None or img_pil is None:
        return {"wrinkle": 75, "spot": 75, "redness": 75, "pore": 75, "texture": 75}
    arr = np.array(img_pil.convert("RGB")).astype("float32")
    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()
    redness = int(np.clip(85 - (r - g) * 0.3, 55, 100))
    texture = int(np.clip(80 - arr.std() * 0.015, 60, 100))
    spot = int(np.clip(78 - (r - b) * 0.15, 55, 100))
    pore = int(np.clip(80 - arr.std() * 0.01, 55, 100))
    wrinkle = int(np.clip(75 - arr.std() * 0.008, 55, 100))
    return {"wrinkle": wrinkle, "spot": spot, "redness": redness, "pore": pore, "texture": texture}


@dataclass
class QualityResult:
    score: int
    brightness: int
    sharpness: int
    framing: int
    tips: str


def quality_check_cv(rgb_img) -> QualityResult:
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    mean_b = int(np.mean(gray))
    bright_score = 100 - int(abs(mean_b - 145) * 1.2)
    bright_score = clamp(bright_score, 0, 100)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    sharp_score = int(clamp((var / 180.0) * 100, 0, 100))

    edges = cv2.Canny(gray, 50, 150)
    cx0, cx1 = int(w * 0.33), int(w * 0.67)
    cy0, cy1 = int(h * 0.33), int(h * 0.67)
    center = edges[cy0:cy1, cx0:cx1]
    framing_ratio = (np.sum(center) + 1) / (np.sum(edges) + 1)
    framing_score = int(clamp((framing_ratio / 0.55) * 100, 0, 100))

    score = int(round(0.35 * bright_score + 0.40 * sharp_score + 0.25 * framing_score))

    tips = []
    if bright_score < 60:
        tips.append("光線不佳：請面向窗戶或白光、避免背光。")
    if sharp_score < 60:
        tips.append("畫面偏糊：擦拭鏡頭、手肘靠桌、對焦臉部。")
    if framing_score < 55:
        tips.append("構圖偏移：臉置中、保持正臉，避免太近或太遠。")
    if not tips:
        tips.append("拍攝品質良好。")

    return QualityResult(score, bright_score, sharp_score, framing_score, " ".join(tips))


# =========================================================
# 3.1) Fusion: 連續特徵改善%（修正「一直 +0%」）+ 拍攝條件可比對性 gate
# =========================================================
def _gray_stats(rgb_img):
    if cv2 is None or np is None or rgb_img is None:
        return {"brightness": 0.0, "contrast": 0.0}
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return {"brightness": float(np.mean(gray)), "contrast": float(np.std(gray))}

def compare_photo_conditions(ref_rgb, now_rgb):
    a = _gray_stats(ref_rgb)
    b = _gray_stats(now_rgb)
    b_delta = abs(a["brightness"] - b["brightness"])
    c_delta = abs(a["contrast"] - b["contrast"])

    comparable = True
    reasons = []
    if b_delta > 35:
        comparable = False
        reasons.append("兩次拍攝亮度差異較大（建議同光源、避免背光）")
    if c_delta > 18:
        comparable = False
        reasons.append("兩次拍攝對比差異較大（建議同距離/同角度）")

    return {"comparable": comparable, "brightness_delta": round(b_delta, 1), "contrast_delta": round(c_delta, 1), "reasons": reasons}

def _resize_for_analysis_rgb(rgb, max_w=700):
    if cv2 is None or np is None or rgb is None:
        return rgb
    h, w = rgb.shape[:2]
    if w <= max_w:
        return rgb
    scale = max_w / float(w)
    new_h = int(h * scale)
    return cv2.resize(rgb, (max_w, new_h), interpolation=cv2.INTER_AREA)

def _central_crop_rgb(rgb, ratio=0.70):
    if rgb is None:
        return rgb
    h, w = rgb.shape[:2]
    ch, cw = int(h * ratio), int(w * ratio)
    y0 = max((h - ch) // 2, 0)
    x0 = max((w - cw) // 2, 0)
    return rgb[y0:y0+ch, x0:x0+cw]

def metric_pack_continuous(rgb_img):
    if cv2 is None or np is None or rgb_img is None:
        return {"texture": 0.0, "spots": 0.0, "pores": 0.0, "smoothness": 0.0}

    rgb = _resize_for_analysis_rgb(rgb_img, max_w=700)
    rgb = _central_crop_rgb(rgb, ratio=0.70)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture = float(np.mean(np.abs(lap)))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    spots = float(np.mean(v < 80.0))  # 0..1

    edges = cv2.Canny(gray, 70, 160)
    pores = float(np.mean(edges > 0))  # 0..1

    bil = cv2.bilateralFilter(gray, d=7, sigmaColor=45, sigmaSpace=45)
    res = gray.astype(np.float32) - bil.astype(np.float32)
    hf = float(np.mean(np.abs(res)))
    smoothness = 1.0 / (1.0 + hf)  # 0..1 higher is smoother

    return {"texture": texture, "spots": spots, "pores": pores, "smoothness": smoothness}

def improvement_pct_float(before: dict, now: dict):
    def _clip(x):
        return float(np.clip(x, -50.0, 50.0)) if np is not None else max(-50.0, min(50.0, x))

    out = {}
    for k in ("texture", "spots", "pores"):
        b = float(before.get(k, 0.0) or 0.0)
        n = float(now.get(k, 0.0) or 0.0)
        out[k] = 0.0 if b <= 1e-9 else _clip((b - n) / b * 100.0)

    b = float(before.get("smoothness", 0.0) or 0.0)
    n = float(now.get("smoothness", 0.0) or 0.0)
    out["smoothness"] = 0.0 if b <= 1e-9 else _clip((n - b) / b * 100.0)
    return out

def fmt_pct_1dp(x):
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


class SkinEngine:
    """Face alignment and skin analysis engine."""

    def _normalize_lighting(self, src_rgb, ref_rgb):
        src = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        sL, sA, sB = cv2.split(src)
        rL, _, _ = cv2.split(ref)

        s_mean, s_std = cv2.meanStdDev(sL)
        r_mean, r_std = cv2.meanStdDev(rL)

        s_mean = float(s_mean[0][0]); s_std = float(s_std[0][0])
        r_mean = float(r_mean[0][0]); r_std = float(r_std[0][0])

        s_std = max(1e-6, s_std)
        r_std = max(1e-6, r_std)

        L = (sL - s_mean) * (r_std / s_std) + r_mean
        L = np.clip(L, 0, 255)

        merged = cv2.merge([L, sA, sB]).astype(np.uint8)
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def align_faces(self, src_img_rgb, ref_img_rgb):
        H, W = ref_img_rgb.shape[:2]
        src_resized = cv2.resize(src_img_rgb, (W, H))

        g1 = cv2.cvtColor(ref_img_rgb, cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(src_resized, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=1200)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)

        if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)
        good = matches[:140]

        if len(good) < 25:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        pts_ref = np.float32([k1[m.queryIdx].pt for m in good])
        pts_src = np.float32([k2[m.trainIdx].pt for m in good])

        M, inliers = cv2.estimateAffinePartial2D(
            pts_src, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None or inliers is None:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, 0.0

        inlier_ratio = float(np.mean(inliers))
        if inlier_ratio < 0.25:
            aligned = self._normalize_lighting(src_resized, ref_img_rgb)
            return aligned, False, inlier_ratio

        aligned = cv2.warpAffine(
            src_resized, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        aligned = self._normalize_lighting(aligned, ref_img_rgb)
        return aligned, True, inlier_ratio

    def analyze(self, rgb_img):
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)

        mean_a = float(np.mean(lab[:, :, 1]))
        red_score = 100 - (mean_a - 128) * 4.0
        redness = clamp(int(red_score), 50, 99)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25, 10
        )
        spot_score = 100 - (float(np.sum(thresh)) / float(thresh.size)) * 200
        spot = clamp(int(spot_score), 55, 98)

        edges = cv2.Canny(gray, 50, 150)
        wrinkle = clamp(int(100 - (float(np.sum(edges)) / float(edges.size)) * 500), 55, 99)
        pore = clamp(int(100 - (float(np.sum(edges)) / float(edges.size)) * 300), 55, 99)
        texture = int(round((wrinkle + pore) / 2))

        return {"wrinkle": wrinkle, "spot": spot, "redness": redness, "pore": pore, "texture": texture}


def compute_confidence(align_success: bool, inlier_ratio: float, q_score: int) -> int:
    base = 55
    base += int(round((q_score - 60) * 0.6))
    if align_success:
        base += 15
        base += int(round(inlier_ratio * 20))
    else:
        base -= 10
    return clamp(base, 10, 98)


def badge_conf(conf: int) -> str:
    if conf >= 80:
        return "badge-ok"
    if conf >= 60:
        return "badge-warn"
    return "badge-bad"


def conf_label(conf: int) -> str:
    if conf >= 80:
        return "可信度高"
    if conf >= 60:
        return "可信度中"
    return "可信度低（建議重拍）"


# =========================================================
# 4) CRUD: users / records / appointments / alerts
# =========================================================
def get_or_create_user(name: str, treatment: str, op_date: str, phone: str = ""):
    conn = db_conn()
    cur = conn.cursor()
    user_id = uuid.uuid4().hex
    cur.execute(
        """
        INSERT INTO users (user_id, phone, name, treatment, op_date, created_at, before_img_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, phone.strip(), name.strip(), treatment.strip(), op_date, _now_str(), None)
    )
    conn.commit()
    conn.close()
    return user_id


def fetch_user(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_users(limit=200):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users ORDER BY COALESCE(created_at,'') DESC, rowid DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_user(user_id: str) -> bool:
    """刪除用戶及其所有相關記錄"""
    conn = db_conn()
    cur = conn.cursor()
    try:
        # 刪除相關記錄
        cur.execute("DELETE FROM records WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM appointments WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM alerts WHERE user_id=?", (user_id,))
        # 刪除用戶
        cur.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        conn.close()


def set_before_img(user_id: str, path: str | None):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("UPDATE users SET before_img_path=? WHERE user_id=?", (path, user_id))
    conn.commit()
    conn.close()


def upsert_record(user_id: str, stage: str, op_date: str | None, img_path: str,
                  q_score: int, confidence: int, metrics: dict, note: str = ""):
    conn = db_conn()
    cur = conn.cursor()

    rec_id = uuid.uuid4().hex
    uploaded_at = _now_str()

    postop_date = None
    try:
        if op_date:
            d = stage_to_days(stage)
            if d is not None:
                base = datetime.strptime(op_date, "%Y-%m-%d").date()
                postop_date = (base + timedelta(days=d)).isoformat()
    except Exception:
        postop_date = None

    record_date = postop_date or date.today().isoformat()

    cur.execute(
        """
        INSERT INTO records (
            id, user_id, stage, record_date, postop_date, uploaded_at, img_path,
            q_score, confidence, wrinkle, spot, redness, pore, texture, note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, stage) DO UPDATE SET
            record_date=excluded.record_date,
            postop_date=excluded.postop_date,
            uploaded_at=excluded.uploaded_at,
            img_path=excluded.img_path,
            q_score=excluded.q_score,
            confidence=excluded.confidence,
            wrinkle=excluded.wrinkle,
            spot=excluded.spot,
            redness=excluded.redness,
            pore=excluded.pore,
            texture=excluded.texture,
            note=excluded.note
        """,
        (
            rec_id, user_id, stage, record_date, postop_date, uploaded_at, img_path,
            int(q_score), int(confidence),
            int(metrics.get("wrinkle", 0)), int(metrics.get("spot", 0)), int(metrics.get("redness", 0)),
            int(metrics.get("pore", 0)), int(metrics.get("texture", 0)), note
        )
    )
    conn.commit()
    conn.close()


def fetch_records(user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM records WHERE user_id=?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    recs = [dict(r) for r in rows]
    recs.sort(key=lambda r: stage_order(r.get("stage", "")))
    return recs


def create_appointment(user_id: str, appt_dt: str, note: str = ""):
    conn = db_conn()
    cur = conn.cursor()
    try:
        appt_id = uuid.uuid4().hex
        cur.execute(
            """
            INSERT INTO appointments (id, user_id, appt_dt, note, status, created_at)
            VALUES (?, ?, ?, ?, 'requested', ?)
            """,
            (appt_id, user_id, appt_dt, note, _now_str())
        )
        conn.commit()
        conn.close()
        return True, "預約已送出（待診所確認）"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "此時段已送出過預約（避免重複）"


def fetch_appointments(user_id: str, limit: int = 100):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM appointments
        WHERE user_id=?
        ORDER BY appt_dt ASC
        LIMIT ?
        """,
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def cancel_appointment(appt_id: str, user_id: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE appointments
        SET status='cancelled'
        WHERE id=? AND user_id=? AND status IN ('requested','confirmed')
        """,
        (appt_id, user_id)
    )
    changed = cur.rowcount
    conn.commit()
    conn.close()
    return changed > 0


def fetch_alerts(limit=50, status_filter=None):
    conn = db_conn()
    cur = conn.cursor()
    
    query = """
        SELECT a.*, u.name, u.treatment
        FROM alerts a
        LEFT JOIN users u ON u.user_id=a.user_id
    """
    params = []
    
    if status_filter:
        if status_filter == "open":
            query += " WHERE (a.status IS NULL OR a.status='open') AND a.resolved=0"
        elif status_filter == "canceled":
            query += " WHERE a.status='canceled'"
        elif status_filter == "closed" or status_filter == "resolved":
            query += " WHERE a.resolved=1 OR a.status='resolved'"
    
    query += " ORDER BY a.created_at DESC LIMIT ?"
    params.append(limit)
    
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def fetch_user_alerts(user_id: str, limit: int = 50):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM alerts
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_or_update_alert(
    user_id: str,
    stage: str,
    severity: str,
    reason: str,
    symptoms: str,
    metrics: dict,
    img_path: str,
    contact_method: str = "站內/文字訊息",
    contact_time: str = "",
    no_call: int = 1,
    user_note: str = ""
):
    """
    防呆策略：
    - 30 分鐘內同一位客戶重複通報 -> 不新增新單，改成更新同一單（追加內容）
    - status=open 且 resolved=0 才視為有效通報
    """
    conn = db_conn()
    cur = conn.cursor()
    now = _now_str()

    cur.execute(
        """
        SELECT id, created_at, symptoms, user_note
        FROM alerts
        WHERE user_id=? AND resolved=0 AND (status IS NULL OR status='open')
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (user_id,)
    )
    row = cur.fetchone()

    def _minutes_diff(t1: str, t2: str) -> float:
        try:
            a = datetime.strptime(t1, "%Y-%m-%d %H:%M:%S")
            b = datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
            return (b - a).total_seconds() / 60.0
        except Exception:
            return 9999.0

    if row and _minutes_diff(row["created_at"], now) <= 30:
        alert_id = row["id"]
        prev_sym = row["symptoms"] or ""
        prev_note = row["user_note"] or ""

        appended_symptoms = prev_sym
        if symptoms and symptoms not in prev_sym:
            appended_symptoms = (prev_sym + "\n" if prev_sym else "") + f"[{now}] {symptoms}"

        appended_note = prev_note
        if user_note and user_note not in prev_note:
            appended_note = (prev_note + "\n" if prev_note else "") + f"[{now}] {user_note}"

        cur.execute(
            """
            UPDATE alerts
            SET stage=?,
                severity=?,
                reason=?,
                symptoms=?,
                metrics_json=?,
                img_path=?,
                updated_at=?,
                contact_method=?,
                contact_time=?,
                no_call=?,
                user_note=?,
                status='open'
            WHERE id=?
            """,
            (
                stage,
                severity,
                reason,
                appended_symptoms,
                json.dumps(metrics, ensure_ascii=False),
                img_path,
                now,
                contact_method,
                contact_time,
                int(no_call),
                appended_note,
                alert_id
            )
        )
        conn.commit()
        conn.close()
        return "updated", alert_id

    alert_id = uuid.uuid4().hex
    cur.execute(
        """
        INSERT INTO alerts (
            id, user_id, created_at, stage, severity, reason, symptoms, metrics_json, img_path,
            resolved, status, updated_at, contact_method, contact_time, no_call, user_note
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'open', ?, ?, ?, ?, ?)
        """,
        (
            alert_id,
            user_id,
            now,
            stage,
            severity,
            reason,
            symptoms,
            json.dumps(metrics, ensure_ascii=False),
            img_path,
            now,
            contact_method,
            contact_time,
            int(no_call),
            user_note
        )
    )
    conn.commit()
    conn.close()
    return "created", alert_id


def cancel_alert(alert_id: str, user_id: str, cancel_reason: str):
    conn = db_conn()
    cur = conn.cursor()
    now = _now_str()
    cur.execute(
        """
        UPDATE alerts
        SET status='canceled',
            canceled_at=?,
            cancel_reason=?,
            updated_at=?
        WHERE id=? AND user_id=? AND resolved=0 AND (status IS NULL OR status='open')
        """,
        (now, cancel_reason, now, alert_id, user_id)
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


def clinic_close_alert(alert_id: str, note: str = ""):
    """診所標記通報為已結案"""
    conn = db_conn()
    cur = conn.cursor()
    now = _now_str()
    cur.execute(
        """
        UPDATE alerts
        SET resolved=1,
            status='resolved',
            updated_at=?,
            user_note=COALESCE(user_note || '\n', '') || ?
        WHERE id=?
        """,
        (now, f"[{now}] 診所備註：{note}" if note.strip() else "", alert_id)
    )
    conn.commit()
    changed = cur.rowcount
    conn.close()
    return changed > 0


# =========================================================
# 5) Nurse suggestion
# =========================================================
def nurse_advice(stage: str, redness_score: int, low_conf: bool):
    severe = False
    advice = []

    if stage in ("術後第 1 天", "術後第 2 天", "術後第 3 天"):
        advice.append("目前屬正常術後反應期：加強保濕、避免高溫環境與劇烈運動。")
        advice.append("建議每 2–3 小時補一次修復保濕，外出務必防曬。")
        if redness_score < 55:
            severe = True
            advice.append("退紅指數偏低：可能反應較強，建議加強冰敷並視情況主動回報。")
    elif stage == "術後第 7 天":
        advice.append("進入代謝/結痂期：請勿摳除，洗臉輕柔，外出加強防曬。")
    elif stage in ("術後第 14 天", "術後第 30 天"):
        advice.append("進入穩定期：持續修復、防曬與作息，能讓成效維持更久。")
    else:
        advice.append("膚況大致穩定：依醫師建議規劃保養型維持療程。")

    if low_conf:
        advice_tip = "本次拍攝條件可能影響判讀：建議在同光源、同距離、同角度重拍以提高準確性。"
        advice.append(advice_tip)

    return severe, advice


# =========================================================
# 6) Charts
# =========================================================
def plot_trend(records):
    if go is None or not records:
        return None
    x = [r.get("stage", "") for r in records]

    avg_scores, reds, confs = [], [], []
    for r in records:
        m = {
            "wrinkle": safe_int(r.get("wrinkle"), 0),
            "spot": safe_int(r.get("spot"), 0),
            "redness": safe_int(r.get("redness"), 0),
            "pore": safe_int(r.get("pore"), 0),
            "texture": safe_int(r.get("texture"), 0),
        }
        avg_scores.append(metrics_avg(m))
        reds.append(m["redness"])
        confs.append(safe_int(r.get("confidence"), 0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=avg_scores, name="綜合分數", mode="lines+markers", 
                           line=dict(color="#2e7d32", width=3),
                           marker=dict(size=10, color=avg_scores, colorscale="RdYlGn", 
                                      colorbar=dict(title="分數"), showscale=True, cmin=0, cmax=100)))
    fig.add_trace(go.Scatter(x=x, y=reds, name="退紅指數", mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Bar(x=x, y=confs, name="可信度", yaxis="y3", opacity=0.35))

    fig.update_layout(
        title="術後恢復趨勢（數值越高代表越進步）",
        height=420,
        xaxis=dict(title="術後階段"),
        yaxis=dict(title="綜合分數", range=[0, 100]),
        yaxis2=dict(title="退紅指數", overlaying="y", side="right", range=[0, 100]),
        yaxis3=dict(title="可信度", anchor="free", overlaying="y", side="right", position=0.95, range=[0, 100]),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode="x unified"
    )
    return fig


def plot_radar(m):
    if go is None or not m:
        return None
    cats = ["紋路", "斑點", "退紅", "毛孔", "平滑"]
    vals = [m["wrinkle"], m["spot"], m["redness"], m["pore"], m["texture"]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals, theta=cats, fill="toself", name="本次"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=260,
        margin=dict(t=20, b=20, l=40, r=40)
    )
    return fig


# =========================================================
# 7) Session / Login
# =========================================================
def ensure_default_user():
    conn = db_conn()
    cur = conn.cursor()
    default_id = "0912345678"
    cur.execute("SELECT user_id FROM users WHERE user_id=?", (default_id,))
    row = cur.fetchone()
    if not row:
        cur.execute(
            """
            INSERT INTO users (user_id, phone, name, treatment, op_date, created_at, before_img_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                default_id,
                default_id,
                "王小美",
                "皮秒雷射 + 蜂巢探頭",
                str(date.today() - timedelta(days=1)),
                _now_str(),
                None,
            ),
        )
    conn.commit()
    conn.close()


ensure_default_user()

if "role" not in st.session_state:
    st.session_state.role = "client"  # client / clinic
if "clinic_authed" not in st.session_state:
    st.session_state.clinic_authed = False


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "alert_confirm_open" not in st.session_state:
    st.session_state.alert_confirm_open = False



# =========================================================
# Clinic dashboard (role-based)
# =========================================================
def render_clinic_dashboard():
    st.subheader("診所端工作台")
    st.caption("預設僅顯示『待處理通報』；已取消與已結案可切換查看，避免淹沒重點。")

    if not st.session_state.get("clinic_authed", False):
        st.warning("請先在左側輸入 PIN 登入診所端。")
        return

    tabA, tabB = st.tabs(["🚑 通報工作清單", "👥 客戶與病歷查詢"])

    with tabA:
        colf = st.columns([1.2, 2.0, 1.2])
        status = colf[0].selectbox("篩選", ["待處理", "已取消", "已結案"], index=0)
        keyword = colf[1].text_input("搜尋（姓名/電話/療程/原因/症狀）", placeholder="輸入關鍵字…")
        limit = colf[2].selectbox("顯示筆數", [20, 50, 100, 200], index=1)

        sf = "open" if status == "待處理" else ("canceled" if status == "已取消" else "closed")
        rows = fetch_alerts(limit=limit, status_filter=sf)

        if keyword.strip():
            k = keyword.strip()
            rows = [
                r for r in rows
                if k in (" ".join([
                    str(r.get("name","")), str(r.get("phone","")), str(r.get("treatment","")),
                    str(r.get("reason","")), str(r.get("symptoms",""))
                ]))
            ]

        if not rows:
            st.info("目前沒有符合條件的通報。")
        else:
            for a in rows:
                status_raw = (a.get("status") or "open").lower()
                resolved = int(a.get("resolved", 0) or 0)
                is_open = (resolved == 0 and status_raw in ("open", ""))

                status_txt = "待處理" if is_open else ("已取消" if status_raw == "canceled" else "已結案")
                sev = a.get("severity", "normal")
                sev_txt = "高" if sev == "high" else "一般"

                st.markdown(
                    f"""
<div class="card">
  <div><b>{a.get('name','(未填姓名)')}</b>｜{a.get('treatment','')}</div>
  <div class="small">狀態：{status_txt}｜時間：{a.get('created_at','')}｜術後階段：{a.get('stage','')}｜嚴重度：{sev_txt}</div>
  <hr/>
  <div><b>原因：</b> {a.get('reason','')}</div>
  <div><b>症狀：</b> {a.get('symptoms','（未填）') if a.get('symptoms') else '（未填）'}</div>
  <div class="small"><b>聯絡偏好：</b> {(a.get('contact_method') or '—')} {'（不希望電話）' if int(a.get('no_call') or 0)==1 else ''} {(('｜方便時段：'+a.get('contact_time')) if a.get('contact_time') else '')}</div>
  <div class="small"><b>客戶補充：</b> {a.get('user_note') or '—'}</div>
  {('<div class="small"><b>取消原因：</b> '+(a.get('cancel_reason') or '—')+'｜取消時間：'+(a.get('canceled_at') or '—')+'</div>') if status_raw=='canceled' else ''}
</div>
""",
                    unsafe_allow_html=True
                )

                if a.get("img_path") and os.path.exists(a["img_path"]):
                    st.image(a["img_path"], caption="通報當下照片", width=420)

                if is_open:
                    with st.expander("處理 / 結案", expanded=False):
                        note = st.text_area("診所備註（可選）", key=f"clinic_note_{a['id']}")
                        if st.button("✅ 標記結案", key=f"close_{a['id']}", use_container_width=True):
                            ok = clinic_close_alert(a["id"], note)
                            st.success("已結案。") if ok else st.warning("結案失敗。")
                            if ok:
                                st.rerun()

    with tabB:
        st.markdown("#### 客戶查詢")
        users = fetch_users(limit=500)
        q = st.text_input("搜尋（姓名/電話/療程）", placeholder="例如：王小美 / 0912 / 皮秒…")
        results = users if not q.strip() else [
            u for u in users if q.strip() in (" ".join([str(u.get("name","")), str(u.get("phone","")), str(u.get("treatment",""))]))
        ]
        if not results:
            st.info("沒有符合條件的客戶。")
        else:
            labels = [f"{u.get('name','')}｜{u.get('phone') or '未填電話'}｜{u.get('treatment','')}" for u in results]
            pick = st.selectbox("選擇客戶", labels, index=0)
            u = results[labels.index(pick)]
            
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{u.get('name','')}**　電話：{u.get('phone') or '未填'}　療程：{u.get('treatment') or '—'}　基準日：{u.get('op_date') or '—'}")
            with col2:
                if st.button("🗑️ 刪除客戶", use_container_width=True):
                    st.session_state.confirm_delete_user_id = u["user_id"]
                    st.session_state.confirm_delete_user_name = u.get("name", "")
            
            st.markdown("---")
            
            # 刪除確認
            if st.session_state.get("confirm_delete_user_id"):
                st.warning(
                    f"⚠️ **確認刪除客戶「{st.session_state.get('confirm_delete_user_name')}」？**\n\n"
                    "此操作將永久刪除該客戶的所有記錄（病歷、預約、通報等），且無法復原。"
                )
                col_yes, col_no, col_blank = st.columns([1, 1, 3])
                with col_yes:
                    if st.button("✅ 確認刪除", use_container_width=True):
                        ok = delete_user(st.session_state.confirm_delete_user_id)
                        if ok:
                            st.success("✅ 客戶已刪除")
                            st.session_state.confirm_delete_user_id = None
                            st.session_state.confirm_delete_user_name = None
                            st.rerun()
                        else:
                            st.error("❌ 刪除失敗，請稍後重試")
                with col_no:
                    if st.button("❌ 取消", use_container_width=True):
                        st.session_state.confirm_delete_user_id = None
                        st.session_state.confirm_delete_user_name = None
                        st.rerun()
            
            recs = fetch_records(u["user_id"])
            if not recs:
                st.info("此客戶尚無病歷。")
            else:
                if go is not None:
                    fig = plot_trend(recs)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown("### 歷史紀錄")
                for r in recs:
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        if r.get("img_path") and os.path.exists(r["img_path"]):
                            st.image(r["img_path"], caption=r.get("stage",""), use_container_width=True)
                    with c2:
                        st.write(f"{r.get('stage','')}｜術後日 {r.get('postop_date') or r.get('record_date','—')}")
                        st.caption(f"上傳：{r.get('uploaded_at') or '—'}｜品質：{safe_int(r.get('q_score'),0)}｜可信度：{safe_int(r.get('confidence'),0)}")

# =========================================================
# 8) UI
# =========================================================
st.markdown(f"## {APP_TITLE}")

# 初始化登入狀態
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "clinic_authed" not in st.session_state:
    st.session_state.clinic_authed = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# 檢查是否已登入
is_customer_logged_in = st.session_state.logged_in and st.session_state.user_id is not None
is_clinic_logged_in = st.session_state.clinic_authed

with st.sidebar:
    st.markdown("### 登入")
    
    if is_customer_logged_in:
        # 客戶已登入
        user = fetch_user(st.session_state.user_id)
        st.markdown(
            f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px;'>
    <div style='font-size: 20px; font-weight: bold; margin-bottom: 8px;'>{user.get('name','')}</div>
    <div style='font-size: 14px; opacity: 0.9;'>療程：{user.get('treatment', '—')}</div>
    <div style='font-size: 14px; opacity: 0.9;'>手術日期：{user.get('op_date', '—')}</div>
</div>
""",
            unsafe_allow_html=True
        )
        if st.button("安全登出", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.alert_confirm_open = False
            st.rerun()
    
    elif is_clinic_logged_in:
        # 診所已登入
        open_alerts = fetch_alerts(limit=200, status_filter="open")
        st.metric("待處理通報", len(open_alerts))
        if st.button("登出診所端", use_container_width=True):
            st.session_state.clinic_authed = False
            st.rerun()
    
    else:
        # 未登入：顯示兩個選項卡
        tab_client, tab_clinic = st.tabs(["👤 客戶端", "🏥 診所端"])
        
        with tab_client:
            st.markdown("### 客戶登入")
            
            users = fetch_users(limit=200)
            display_users = ["—"] + [f'{u.get("name","(未命名)") }｜{u.get("treatment","")}' for u in users]
            
            st.markdown("#### 既有客戶")
            existing = st.selectbox("選擇客戶", display_users, index=0, label_visibility="collapsed", key="sidebar_existing_user")
            if st.button("登入", use_container_width=True, key="sidebar_login_existing"):
                if existing != "—":
                    idx = display_users.index(existing) - 1
                    st.session_state.logged_in = True
                    st.session_state.user_id = users[idx]["user_id"]
                    st.rerun()
                else:
                    st.warning("請先選擇客戶")
            
            st.markdown("---")
            st.markdown("#### 新增客戶")
            new_name = st.text_input("姓名", "", label_visibility="collapsed", placeholder="客戶姓名", key="sidebar_new_name")
            new_phone = st.text_input("電話", "", label_visibility="collapsed", placeholder="0912xxxxxx（選填）", key="sidebar_new_phone")
            new_treatment = st.text_input("療程", "音波拉提 + 水光針", label_visibility="collapsed", placeholder="例如：皮秒雷射", key="sidebar_new_treatment")
            
            if st.button("建立並登入", use_container_width=True, key="sidebar_create_and_login"):
                if new_name.strip():
                    uid = get_or_create_user(new_name.strip(), new_treatment.strip(), date.today().isoformat(), phone=new_phone.strip())
                    st.session_state.logged_in = True
                    st.session_state.user_id = uid
                    st.rerun()
                else:
                    st.warning("請填寫姓名")
        
        with tab_clinic:
            st.markdown("### 診所端登入")
            pin = st.text_input("PIN", type="password", placeholder="預設 1234", key="sidebar_clinic_pin")
            if st.button("登入", use_container_width=True, key="sidebar_clinic_login"):
                if pin == CLINIC_PIN:
                    st.session_state.clinic_authed = True
                    st.toast("✅ 已登入診所端")
                    st.rerun()
                else:
                    st.error("PIN 錯誤")

# =========================================================
# Role routing (執行於 sidebar 之後)
# =========================================================
if is_clinic_logged_in:
    render_clinic_dashboard()

if not is_customer_logged_in:
    st.stop()

user = fetch_user(st.session_state.user_id)
if not user:
    st.error("使用者不存在（資料庫可能損毀或 user_id 不存在）")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🩺 術後追蹤", "📊 成效報告", "📅 預約回診", "🏥 診所通報"])


# -----------------------------------------------------
# Tab1: Post-op tracking
# -----------------------------------------------------
with tab1:
    st.markdown(
        """
<div class="hint">
<b>拍攝指引（讓成效更準、也更像真實醫療服務）</b><br/>
1) 面向窗戶或白光、避免背光　2) 正臉、眼睛水平　3) 距離約 30–40 cm　4) 不用濾鏡/美肌　5) 背景盡量純色
</div>
""",
        unsafe_allow_html=True,
    )

    op_date = user.get("op_date") or ""
    st.markdown(f"**手術日期（基準）**：{op_date or '—'}")

    cA, cB, cC = st.columns([1.6, 1.8, 1.2])

    stage = cA.selectbox("術後階段", STAGES, index=0)
    record_note = cA.text_input("備註（選填）", placeholder="例如：今天有上修復霜/戶外曝曬...")

    postop_date = None
    d = stage_to_days(stage)
    if op_date and d is not None:
        try:
            base = datetime.strptime(op_date, "%Y-%m-%d").date()
            postop_date = (base + timedelta(days=d)).isoformat()
        except Exception:
            postop_date = None

    cA.markdown(
        f"""
<div class="card">
  <div><b>術後階段：</b> {stage}</div>
  <div><b>術後日（推算）：</b> {postop_date or "（此階段為 30 天以上／非固定日）"}</div>
  <div class="small">晚幾天才上傳也可：系統仍以術後日歸檔，同時保留實際上傳時間。</div>
</div>
""",
        unsafe_allow_html=True
    )

    curr_file = cB.file_uploader("上傳今日照片（正臉）", type=["jpg", "jpeg", "png"])

    cC.markdown("**術前照片（Baseline）**")
    before_path = user.get("before_img_path")
    if before_path and os.path.exists(before_path):
        cC.success("✅ 已鎖定術前圖")
        if cC.button("重新上傳術前圖", use_container_width=True):
            set_before_img(user["user_id"], None)
            st.rerun()
        before_file = None
    else:
        before_file = cC.file_uploader("上傳術前照片", type=["jpg", "jpeg", "png"])

    img_ref = load_image_rgb(before_path) if (before_path and os.path.exists(before_path)) else load_image_rgb(before_file)
    if (not before_path or not os.path.exists(before_path)) and img_ref is not None and Image is not None:
        path = save_rgb_image(img_ref, prefix=f"before_{user['user_id']}")
        set_before_img(user["user_id"], path)
        user = fetch_user(st.session_state.user_id)
        st.toast("✅ 術前圖已鎖定")

    st.markdown("---")

    if curr_file is None:
        st.warning("📸 請上傳今日照片以進行 AI 分析。若也上傳術前照片，系統會自動生成前後對比與改善%。")
    else:
        curr_pil = Image.open(curr_file).convert("RGB") if Image is not None else None
        img_curr = load_image_rgb(curr_file) if curr_file else None

        use_cv = (np is not None and cv2 is not None and Image is not None and img_curr is not None and img_ref is not None)

        aligned_preview_path = ""
        q_score = 70
        q_detail = None
        conf = 70

        if use_cv:
            engine = SkinEngine()
            with st.spinner("AI 分析中（含校正/對齊）..."):
                aligned, align_ok, inlier_ratio = engine.align_faces(img_curr, img_ref)
                q_detail = quality_check_cv(aligned)
                q_score = int(q_detail.score)
                conf = int(compute_confidence(align_ok, inlier_ratio, q_score))

                base_metrics = engine.analyze(img_ref)
                curr_metrics = engine.analyze(aligned)

            aligned_preview_path = save_rgb_image(aligned, prefix=f"rec_{user['user_id']}")


        st.markdown("### 1) 前後對比")
        
        # 左右並排佈局：照片 vs 護理建議
        col_photo, col_info = st.columns([7,5])
        
        # 定義症狀變數（在兩個列中都能使用）
        sym_red = False
        sym_pain = False
        sym_ooze = False
        sym_swelling = False
        sym_note = ""
        
        with col_photo:
            st.markdown("<div class='photo-card'>", unsafe_allow_html=True)
            if image_comparison is not None:
                image_comparison(img1=img_ref, img2=aligned, label1="術前", label2="目前（已校正）", width=600, in_memory=True)
            else:
                st.image(img_ref, caption="術前", use_container_width=True)
                st.image(aligned, caption="目前（已校正）", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_info:
            st.markdown("#### 症狀回報")
            
            # 症狀程度滑塊（0-10分）
            st.markdown("**紅 / 熱**")
            sym_red_score = st.slider("", 0, 10, 0, key="sym_red_slider", label_visibility="collapsed")
            sym_red = sym_red_score > 0
            
            st.markdown("**痛感明顯**")
            sym_pain_score = st.slider("", 0, 10, 0, key="sym_pain_slider", label_visibility="collapsed")
            sym_pain = sym_pain_score > 0
            
            st.markdown("**滲液 / 不明分泌物**")
            sym_ooze_score = st.slider("", 0, 10, 0, key="sym_ooze_slider", label_visibility="collapsed")
            sym_ooze = sym_ooze_score > 0
            
            st.markdown("**腫脹明顯**")
            sym_swelling_score = st.slider("", 0, 10, 0, key="sym_swelling_slider", label_visibility="collapsed")
            sym_swelling = sym_swelling_score > 0
            
            st.markdown("**補充描述**")
            sym_note = st.text_area("", placeholder="例如：下午開始刺痛...", key="col_sym_note", height=50, label_visibility="collapsed")
            
            st.markdown("---")
            st.markdown("#### 護理任務清單")
            
            # 可勾選的護理清單
            task1 = st.checkbox("清潔 10-15 分鐘（每 2-3 小時）", key="task1")
            task2 = st.checkbox("加強保濕（至少 3 次）", key="task2")
            task3 = st.checkbox("避免曬曬、劇烈運動", key="task3")
            task4 = st.checkbox("避免辛辣刺激飲食", key="task4")
            task5 = st.checkbox("避免接觸、去角質", key="task5")
            task6 = st.checkbox("外出防曬（SPF30+）", key="task6")
            
            # 計算完成度
            completed_tasks = sum([task1, task2, task3, task4, task5, task6])
            completion_rate = int((completed_tasks / 6) * 100)
            
            st.markdown("---")
            st.metric("今日完成度", f"{completion_rate}%")

        # 組合症狀文本（供後續使用）
        symptoms_list = []
        if sym_red: symptoms_list.append("紅/熱")
        if sym_pain: symptoms_list.append("痛感")
        if sym_ooze: symptoms_list.append("滲液")
        if sym_swelling: symptoms_list.append("腫脹")
        if sym_note.strip(): symptoms_list.append("備註：" + sym_note.strip())
        symptoms_text = "；".join(symptoms_list) if symptoms_list else ""

        base_avg = metrics_avg(base_metrics)
        curr_avg = metrics_avg(curr_metrics)

        low_conf = (conf < 60) or (q_score < 55)
        pct_tag = "（建議同光源重拍）" if low_conf else ""

        st.markdown("---")
        st.markdown("### 2) 拍攝品質與可信度")
        b_class = badge_conf(conf)
        st.markdown(
            f"""
<div class="card">
  <div><b>拍攝品質：</b> {q_score}/100　<span class="small">(亮度 {q_detail.brightness}｜清晰 {q_detail.sharpness}｜構圖 {q_detail.framing})</span></div>
  <div class="small">{q_detail.tips}</div>
  <hr/>
  <div><b>分析可信度：</b> <span class="{b_class}">{conf_label(conf)}（{conf}/100）</span></div>
  <div class="small">可信度低仍顯示改善%，但會加註提醒，避免誤判。</div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown("---")
        
        # --- Fusion: continuous improvement (fix +0%) + comparability gate
        cond = compare_photo_conditions(img_ref, aligned)
        if not cond["comparable"]:
            st.warning("拍攝條件差異較大：不建議直接比較改善%。建議同光源/同距離/同角度重拍後再看趨勢。")
            for r in cond["reasons"]:
                st.write("• " + r)
            pct_tag = "（不建議判讀）"
            low_conf = True

        m_base_c = metric_pack_continuous(img_ref)
        m_now_c = metric_pack_continuous(aligned)
        imp_c = improvement_pct_float(m_base_c, m_now_c)
        comp_impr = float(np.mean([imp_c["texture"], imp_c["spots"], imp_c["pores"], imp_c["smoothness"]])) if np is not None else (
            (imp_c["texture"] + imp_c["spots"] + imp_c["pores"] + imp_c["smoothness"]) / 4.0
        )

        st.markdown("---")
        st.markdown("### 3) 成效摘要（客人最有感）")
        red_impr = improvement_pct(curr_metrics["redness"], base_metrics["redness"])
        st.markdown(
            f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-title">綜合趨勢（連續特徵）</div>
    <div class="metric-val">{fmt_pct_1dp(comp_impr)}</div>
    <div class="metric-sub">依拍攝條件不同可能有波動 {pct_tag}</div>
  </div>
  <div class="metric-box">
    <div class="metric-title">退紅指數（0–100）</div>
    <div class="metric-val">{curr_metrics['redness']}/100</div>
    <div class="metric-sub">改善：{red_impr:+d}% {pct_tag}</div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

        st.markdown("### 4) 分項改善（%）")
        baseline_missing = (img_ref is None)
        if baseline_missing:
            st.info("尚未鎖定術前照片（Baseline），因此無法計算改善%。請先在右側上傳並鎖定術前照片。")

        if low_conf:
            st.warning("本次照片條件/可信度可能影響精準度：改善%仍顯示，但建議依拍攝指引重拍一次以提高可信度。")

        if not baseline_missing:
            st.write(f"紋路（Texture）：{fmt_pct_1dp(imp_c['texture'])} {pct_tag}")
            st.write(f"斑點（Spots）：{fmt_pct_1dp(imp_c['spots'])} {pct_tag}")
            st.write(f"毛孔（Pores）：{fmt_pct_1dp(imp_c['pores'])} {pct_tag}")
            st.write(f"平滑（Smoothness）：{fmt_pct_1dp(imp_c['smoothness'])} {pct_tag}")

        # 計算風險等級和護理建議
        low_conf_flag = (conf < 60) or (q_score < 55)
        severe_flag, advice_lines = nurse_advice(stage, int(curr_metrics.get("redness", 70)), low_conf_flag)
        auto_bad = bool(severe_flag or sym_ooze or (sym_pain and sym_red) or (int(curr_metrics.get("redness", 100)) < 55))
        risk_label = "🔴 紅燈" if auto_bad else ("🟡 黃燈" if (sym_red or sym_pain or low_conf_flag) else "🟢 綠燈")

        st.markdown("### 5) AI 護理師建議")
        
        # 根據風險等級設定卡片顏色
        if auto_bad:
            card_color = "#fff3cd"  # 黃色背景（警告）
            border_color = "#ff6b6b"  # 紅色邊框
        elif (sym_red or sym_pain or low_conf_flag):
            card_color = "#fff9e6"  # 淡黃色背景
            border_color = "#ffa500"  # 橙色邊框
        else:
            card_color = "#e8f5e9"  # 淡綠色背景
            border_color = "#4caf50"  # 綠色邊框
        
        st.markdown(f"""
<div style='border-left: 4px solid {border_color}; background-color: {card_color}; padding: 16px; border-radius: 8px; margin-bottom: 12px;'>
    <div style='font-size: 18px; font-weight: bold; margin-bottom: 12px;'>{risk_label}</div>
    <div style='font-size: 14px; line-height: 1.8;'>
""", unsafe_allow_html=True)
        
        for s in advice_lines:
            st.markdown(f"• {str(s)}", unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 6) 存入病歷")
        save_confirm = st.checkbox("我確認：這是我要存入的照片與術後階段（同一階段會覆蓋更新）", value=False)
        save_btn = st.button("💾 存入病歷（更新本階段）", type="primary", use_container_width=True, disabled=not save_confirm)

        if save_btn:
            upsert_record(
                user_id=user["user_id"],
                stage=stage,
                op_date=user.get("op_date"),
                img_path=aligned_preview_path,
                q_score=int(q_score),
                confidence=int(conf),
                metrics=curr_metrics,
                note=(record_note.strip() or "；".join(advice_lines[:2])),
            )
            st.toast("✅ 已存入病歷（同階段已更新，不會重複）")
            time.sleep(0.2)
            st.rerun()

        st.markdown("---")
        st.markdown("### 7) 通報診所")

        default_reason = "系統判定狀況可能不理想" if auto_bad else "客人主動通報"
        st.markdown(
            f"""
<div class="card">
  <div><b>建議狀態：</b> {'🔴 建議盡快聯絡診所' if auto_bad else '🟢 多半屬可觀察範圍（仍可通報）'}</div>
  <div><b>預設通報理由：</b> {default_reason}</div>
  <div class="small">通報採用「二次確認 + 必填原因 + 30 分鐘內合併節流」。送出後可在「🏥 診所通報」分頁隨時取消。</div>
</div>
""",
            unsafe_allow_html=True
        )

        if st.button("📣 我要通報診所（進入確認）", use_container_width=True):
            st.session_state.alert_confirm_open = True
            st.rerun()

        if st.session_state.alert_confirm_open:
            with st.expander("通報確認（請填寫原因與聯絡偏好）", expanded=True):
                st.markdown("#### 1) 通報原因（必填）")
                reason_choice = st.radio(
                    "請選擇最符合的原因",
                    [
                        "痛感突然變強 / 明顯不適",
                        "紅腫擴大 / 發熱",
                        "疑似滲液 / 結痂異常",
                        "擔心左右不對稱",
                        "我不確定是否正常（想確認）",
                        "其他",
                    ],
                    index=1 if auto_bad else 4,
                )
                extra_note = st.text_area("補充說明（可選）", placeholder="例如：從何時開始、是否逐漸加重、是否影響睡眠…")

                st.markdown("#### 2) 希望診所如何聯絡你（降低打擾）")
                no_call = st.checkbox("我不希望接到電話（偏好文字即可）", value=True)
                contact_method = st.selectbox(
                    "聯絡方式偏好",
                    ["站內/文字訊息", "電話", "電話 + 文字"],
                    index=0
                )
                if no_call and contact_method != "站內/文字訊息":
                    st.info("你已勾選不希望電話聯絡，系統將改以文字訊息為主。")
                    contact_method = "站內/文字訊息"

                contact_time = st.text_input("方便聯絡時段（可選）", placeholder="例如：平日 18:00 後、午休時間、任何時間皆可")

                st.markdown("#### 3) 最終確認")
                st.caption("提醒：30 分鐘內重複通報會自動合併為同一則通報（追加內容），避免診所被多筆通報干擾。")
                colA, colB = st.columns(2)
                with colA:
                    confirm_send = st.button("✅ 確認送出通報", use_container_width=True)
                with colB:
                    cancel_flow = st.button("⬅️ 先不要通報", use_container_width=True)

                if cancel_flow:
                    st.session_state.alert_confirm_open = False
                    st.rerun()

                if confirm_send:
                    severity = "high" if auto_bad else "normal"
                    reason = f"{default_reason}｜{reason_choice}"

                    status, alert_id = create_or_update_alert(
                        user_id=user["user_id"],
                        stage=stage,
                        severity=severity,
                        reason=reason,
                        symptoms=symptoms_text,
                        metrics=curr_metrics,
                        img_path=aligned_preview_path,
                        contact_method=contact_method,
                        contact_time=contact_time,
                        no_call=1 if no_call else 0,
                        user_note=extra_note.strip()
                    )

                    st.session_state.alert_confirm_open = False

                    if status == "updated":
                        st.success("已更新既有通報（已合併新增資訊），診所將以你偏好的方式處理。")
                    else:
                        st.success("已送出通報，診所將以你偏好的方式處理。")

                    st.info("若你稍後確認是正常現象，可到「🏥 診所通報」分頁隨時取消，避免診所再打擾你。")


# -----------------------------------------------------
# Tab2: Report / History
# -----------------------------------------------------
with tab2:
    st.subheader("成效報告")
    recs = fetch_records(user["user_id"])
    if not recs:
        st.info("尚無病歷資料。請在「術後追蹤」存入至少一筆。")
    else:
        if go is not None:
            fig = plot_trend(recs)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly 未安裝，將不顯示圖表。")

        st.markdown("---")
        st.markdown("### 歷史紀錄")
        for r in recs:
            m = {
                "wrinkle": safe_int(r.get("wrinkle"), 0),
                "spot": safe_int(r.get("spot"), 0),
                "redness": safe_int(r.get("redness"), 0),
                "pore": safe_int(r.get("pore"), 0),
                "texture": safe_int(r.get("texture"), 0),
            }

            col1, col2 = st.columns([1.0, 1.2])
            with col1:
                if r.get("img_path") and os.path.exists(r["img_path"]):
                    st.image(
                        r["img_path"],
                        caption=f"{r.get('stage','')}｜術後日 {r.get('postop_date') or r.get('record_date','—')}｜上傳 {r.get('uploaded_at') or '—'}",
                        use_container_width=True
                    )
            with col2:
                st.markdown(f"**{r.get('stage','')}｜術後日 {r.get('postop_date') or r.get('record_date','—')}**")
                st.caption(f"上傳時間：{r.get('uploaded_at', '') or '—'}")
                st.caption(f"拍攝品質：{safe_int(r.get('q_score'),0)} / 可信度：{safe_int(r.get('confidence'),0)}")
                if go is not None:
                    radar = plot_radar(m)
                    if radar is not None:
                        st.plotly_chart(radar, use_container_width=True)
                else:
                    st.json(m)
                if r.get("note"):
                    st.caption("備註：" + str(r["note"]))


# -----------------------------------------------------
# Tab3: Appointment
# -----------------------------------------------------
with tab3:
    st.subheader("預約回診")

    today = date.today()
    end_next_year = date(today.year + 1, 12, 31)

    d = st.date_input(
        "日期（不可選今天以前；僅今年~明年）",
        value=today + timedelta(days=7),
        min_value=today,
        max_value=end_next_year
    )

    slots = [
        "10:00", "10:30", "11:00", "11:30",
        "14:00", "14:30", "15:00", "15:30",
        "16:00", "16:30", "17:00"
    ]
    t = st.selectbox("時段（下拉選擇）", slots, index=0)
    note = st.text_input("備註（選填）", value="術後追蹤回診")

    appt_dt = f"{d.isoformat()} {t}"

    confirm_send = st.checkbox("我確認送出此預約時段", value=False)
    if st.button("送出預約", type="primary", use_container_width=True, disabled=not confirm_send):
        ok, msg = create_appointment(user["user_id"], appt_dt, note.strip())
        if ok:
            st.success(msg)
            time.sleep(0.2)
            st.rerun()
        else:
            st.warning(msg)

    st.markdown("---")
    st.markdown("#### 我的預約清單")

    appts_all = fetch_appointments(user["user_id"], limit=100)
    appts = [a for a in appts_all if (a.get("status") or "requested") in ("requested", "confirmed")]

    if not appts:
        st.info("目前沒有有效預約。")
    else:
        for a in appts:
            c1, c2, c3 = st.columns([2.7, 1.1, 1.2])
            c1.write(f"🗓️ {a.get('appt_dt','')} | 備註：{a.get('note','')}")
            c2.write(f"狀態：**{a.get('status','requested')}**")
            confirm = c3.checkbox("確認取消", key=f"confirm_appt_{a['id']}")
            if c3.button("取消預約", key=f"cancel_appt_{a['id']}", disabled=not confirm):
                ok = cancel_appointment(a["id"], user["user_id"])
                if ok:
                    st.toast("已取消預約")
                    time.sleep(0.2)
                    st.rerun()
                else:
                    st.warning("取消失敗")


# -----------------------------------------------------
# Tab4: Alerts (client view + clinic view)
# -----------------------------------------------------
# -----------------------------------------------------
# Tab4: Alerts (client view only; cancelled archived)
# -----------------------------------------------------
with tab4:
    st.subheader("我的通報")
    st.caption("主畫面預設只顯示『有效通報』；已取消可收納避免資訊轟炸。")

    my_alerts = fetch_user_alerts(user["user_id"], limit=200)
    if not my_alerts:
        st.info("你目前沒有通報紀錄。")
    else:
        open_items, canceled_items, closed_items = [], [], []
        for a in my_alerts:
            status = (a.get("status") or "open").lower()
            resolved = int(a.get("resolved", 0) or 0)
            if resolved == 0 and status in ("open", ""):
                open_items.append(a)
            elif status == "canceled":
                canceled_items.append(a)
            else:
                closed_items.append(a)

        st.markdown("### 目前有效通報")
        if not open_items:
            st.write("目前沒有有效通報。")
        else:
            for a in open_items:
                sev = a.get("severity", "normal")
                sev_txt = "高" if sev == "high" else "一般"
                st.markdown(
                    f"""
<div class="card">
  <div><b>通報狀態：</b> 🟢 已送出（未結案）</div>
  <div class="small">送出時間：{a.get('created_at','')}｜更新時間：{a.get('updated_at') or a.get('created_at','')}｜嚴重度：{sev_txt}</div>
  <div class="small">術後階段：{a.get('stage','')}</div>
  <hr/>
  <div><b>原因：</b> {a.get('reason','')}</div>
  <div><b>症狀：</b> {a.get('symptoms','（未填）') if a.get('symptoms') else '（未填）'}</div>
  <div class="small"><b>聯絡偏好：</b> {(a.get('contact_method') or '—')} {'（不希望電話）' if int(a.get('no_call') or 0)==1 else ''} {(('｜方便時段：'+a.get('contact_time')) if a.get('contact_time') else '')}</div>
</div>
""",
                    unsafe_allow_html=True
                )

                with st.expander("取消此通報（不想讓診所再關切）", expanded=False):
                    cancel_reason = st.selectbox(
                        "取消原因（必填）",
                        ["我已確認是正常現象，不需要聯絡", "我誤觸按鈕", "我改用其他方式聯絡診所", "其他"],
                        key=f"cancel_reason_{a.get('id')}"
                    )
                    cancel_note = st.text_input("補充（可選）", key=f"cancel_note_{a.get('id')}", placeholder="例如：已自行冰敷改善、已詢問護理師確認…")
                    if st.button("🧾 確認取消通報", key=f"btn_cancel_{a.get('id')}", use_container_width=True):
                        final_reason = cancel_reason if cancel_reason != "其他" else ("其他：" + (cancel_note.strip() or "未填"))
                        if cancel_reason != "其他" and cancel_note.strip():
                            final_reason = f"{cancel_reason}｜{cancel_note.strip()}"
                        ok = cancel_alert(a.get("id"), user["user_id"], final_reason)
                        st.success("已取消通報。") if ok else st.warning("取消失敗：此通報可能已被診所結案或已取消。")
                        if ok:
                            st.rerun()

        st.markdown("---")
        st.markdown("### 已取消（收納）")
        show_cancelled = st.toggle("顯示已取消紀錄", value=False)
        if show_cancelled:
            if not canceled_items:
                st.write("目前沒有已取消紀錄。")
            else:
                for a in canceled_items[:30]:
                    st.markdown(
                        f"""
<div class="card">
  <div><b>通報狀態：</b> ⚪ 已取消</div>
  <div class="small">送出：{a.get('created_at','')}｜取消：{a.get('canceled_at') or '—'}</div>
  <div class="small"><b>取消原因：</b> {a.get('cancel_reason') or '—'}</div>
</div>
""",
                        unsafe_allow_html=True
                    )

        st.markdown("---")
        st.markdown("### 已結案")
        if not closed_items:
            st.write("目前沒有已結案通報。")
        else:
            with st.expander(f"查看已結案（{len(closed_items)}）", expanded=False):
                for a in closed_items[:30]:
                    st.markdown(
                        f"""
<div class="card">
  <div><b>通報狀態：</b> ✅ 已結案</div>
  <div class="small">送出：{a.get('created_at','')}｜更新：{a.get('updated_at') or a.get('created_at','')}</div>
  <div><b>原因：</b> {a.get('reason','')}</div>
</div>
""",
                        unsafe_allow_html=True
                    )
import streamlit as st

import os, time, json, uuid

from datetime import date, timedelta, datetime

from pathlib import Path



# optional libs

try:

    import plotly.graph_objects as go

    import cv2

    import numpy as np

    from PIL import Image

    from streamlit_image_comparison import image_comparison

except Exception:

    pass



# =========================================================

# 0) 基本設定與資料夾

# =========================================================

APP_TITLE = "美麗追蹤者 Beauty Tracker"

DATA_DIR = Path("user_data")

DATA_DIR.mkdir(exist_ok=True)



st.set_page_config(page_title=APP_TITLE, layout="wide")



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }

.stApp { background-color: #fcfcfc; }

.nurse-box { border: 1px solid #e0e0e0; border-radius: 12px; padding: 14px; background-color: white; }

.metric-title { font-size: 13px; color: #555; margin-top: 4px; }

.metric-val { font-size: 26px; font-weight: 800; color: #222; line-height: 1.1; }

.metric-sub { font-size: 12px; color: #666; }

.pill { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; }

.pill-good { background: #e8f5e9; color: #1b5e20; }

.pill-warn { background: #fff8e1; color: #e65100; }

.pill-bad  { background: #ffebee; color: #b71c1c; }

.hr { height:1px; background:#eee; margin: 12px 0; }

small { color:#666; }

#MainMenu {visibility: hidden;} footer {visibility: hidden;}

</style>

""", unsafe_allow_html=True)



# =========================================================

# 1) 測試用單一使用者（專題 demo）

# =========================================================

DEMO_USER = {

    "phone": "0912345678",

    "name": "王小美 (VIP)",

    "id": "A123456789",

    "treatment": "皮秒雷射 + 蜂巢探頭",

    "op_date": date.today() - timedelta(days=1),

}



# session

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False

if "user_key" not in st.session_state:

    st.session_state.user_key = None



# =========================================================

# 2) 工具：檔案與紀錄

# =========================================================

def user_dir(user_key: str) -> Path:

    d = DATA_DIR / user_key

    d.mkdir(exist_ok=True, parents=True)

    (d / "records").mkdir(exist_ok=True, parents=True)

    return d



def paths(user_key: str):

    d = user_dir(user_key)

    return {

        "root": d,

        "before_img": d / "before.jpg",

        "history_json": d / "history.json",

        "records_dir": d / "records",

    }



def load_history(user_key: str):

    p = paths(user_key)["history_json"]

    if not p.exists():

        return []

    try:

        return json.loads(p.read_text(encoding="utf-8"))

    except Exception:

        return []



def save_history(user_key: str, history: list):

    p = paths(user_key)["history_json"]

    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")



def load_image(file_or_path):

    if file_or_path is None:

        return None

    if isinstance(file_or_path, (str, Path)):

        fp = str(file_or_path)

        if not os.path.exists(fp):

            return None

        img = Image.open(fp).convert("RGB")

    else:

        img = Image.open(file_or_path).convert("RGB")

    return np.array(img)



def save_rgb_image(arr_rgb, dst_path: Path):

    Image.fromarray(arr_rgb).save(str(dst_path))



# =========================================================

# 3) 核心演算法：分析 + 品質檢查 + 改善%

# =========================================================

class SkinEngine:

    def __init__(self):

        pass



    def align_faces(self, src_img, ref_img):

        # demo 先用 resize 對齊，並用品質檢查減少亂跳

        h, w = ref_img.shape[:2]

        return cv2.resize(src_img, (w, h)), True



    def analyze(self, image_rgb):

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)



        # redness: L*a*b a-channel mean

        mean_a = float(np.mean(lab[:, :, 1]))

        red_score = 100 - (mean_a - 128) * 4.0

        redness = int(max(20, min(99, red_score)))



        # spots: adaptive threshold area

        thresh = cv2.adaptiveThreshold(

            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

            cv2.THRESH_BINARY_INV, 25, 10

        )

        spot_score = 100 - (np.sum(thresh) / thresh.size) * 200

        spot = int(max(40, min(95, spot_score)))



        # texture: edges as proxy (wrinkle/pore)

        edges = cv2.Canny(gray, 50, 150)

        wrinkle = float(max(50, 100 - (np.sum(edges) / edges.size) * 500))

        pore = float(max(50, 100 - (np.sum(edges) / edges.size) * 300))

        texture = float((wrinkle + pore) / 2)



        # spot visualization overlay

        vis_spot = image_rgb.copy()

        vis_spot[thresh > 0] = [220, 0, 0]

        vis_spot = cv2.addWeighted(vis_spot, 0.30, image_rgb, 0.70, 0)



        metrics = {

            "wrinkle": int(wrinkle),

            "spot": int(spot),

            "redness": int(redness),

            "pore": int(pore),

            "texture": int(texture),

        }

        return {"metrics": metrics, "vis_spot": vis_spot}



def quality_check(image_rgb):

    """

    回傳 quality dict：

    - ok: 是否允許計算改善%

    - score: 0-100

    - tags: 問題列表

    """

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)



    # brightness

    mean_b = float(np.mean(gray))

    # blur: Laplacian variance

    lap = cv2.Laplacian(gray, cv2.CV_64F)

    blur_var = float(lap.var())



    tags = []

    score = 100.0



    # brightness penalties

    if mean_b < 70:

        tags.append("太暗")

        score -= min(30, (70 - mean_b) * 0.6)

    if mean_b > 185:

        tags.append("太亮/過曝")

        score -= min(30, (mean_b - 185) * 0.6)



    # blur penalties

    # 門檻值可依你實測調整：手機清晰照通常 > 100~200

    if blur_var < 80:

        tags.append("偏模糊")

        score -= min(35, (80 - blur_var) * 0.4)



    score = max(0.0, min(100.0, score))

    ok = score >= 60 and ("太暗" not in tags) and ("太亮/過曝" not in tags)



    return {"ok": ok, "score": int(score), "brightness": int(mean_b), "sharpness": int(blur_var), "tags": tags}



def improvement_pct(curr_score: int, base_score: int):

    """

    對「越高越好」的指標：

    改善% = (curr - base) / (100 - base) * 100

    base=100 時避免除零

    """

    base = max(0, min(100, int(base_score)))

    curr = max(0, min(100, int(curr_score)))

    denom = max(1, 100 - base)

    pct = (curr - base) / denom * 100.0

    # 可允許負值（代表變差），但限制範圍讓 UI 好看

    return float(max(-100.0, min(100.0, pct)))



def metrics_avg(metrics: dict):

    return int(sum(metrics.values()) / max(1, len(metrics)))



# =========================================================

# 4) 圖表

# =========================================================

def plot_trend(history):

    # 排序：先用 stage_day，再用日期

    def key_fn(r):

        sd = int(r.get("stage_day", 999))

        d = r.get("record_date", "9999-12-31")

        return (sd, d)



    hist = sorted(history, key=key_fn)

    labels = [h["stage_label"] for h in hist]

    avg_scores = [int(h.get("avg", 0)) for h in hist]

    reds = [int(h["metrics"]["redness"]) for h in hist]



    fig = go.Figure()

    fig.add_trace(go.Scatter(x=labels, y=avg_scores, name="綜合評分",

                             line=dict(color="#d4af37", width=5),

                             mode="lines+markers"))

    fig.add_trace(go.Scatter(x=labels, y=reds, name="退紅指數",

                             line=dict(color="#e74c3c", width=3, dash="dot"),

                             mode="lines+markers", yaxis="y2"))



    fig.update_layout(

        title="<b>術後恢復趨勢</b>",

        xaxis=dict(title="術後階段", showgrid=False),

        yaxis=dict(title="分數 (越高越好)", range=[0, 100], showgrid=True, gridcolor="#eee"),

        yaxis2=dict(title="退紅指數", overlaying="y", side="right", range=[0, 100], showgrid=False),

        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),

        height=320, margin=dict(l=20, r=20, t=60, b=20),

        hovermode="x unified",

        plot_bgcolor="white", paper_bgcolor="white"

    )

    return fig



def plot_radar(curr):

    cats = ["紋路", "斑點", "退紅度", "毛孔", "平滑"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(

        r=list(curr.values()), theta=cats, fill="toself",

        name="本次", line_color="#d4af37"

    ))

    fig.update_layout(

        polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor="rgba(0,0,0,0)"),

        dragmode=False, height=240, margin=dict(t=20, b=20, l=40, r=40),

        showlegend=False, paper_bgcolor="rgba(0,0,0,0)"

    )

    return fig



# =========================================================

# 5) 術後關懷：任務清單 + 症狀分級 + 护理师SOP

# =========================================================

STAGES = [

    ("術後第 1 天", 1),

    ("術後第 2 天", 2),

    ("術後第 3 天", 3),

    ("術後第 7 天", 7),

    ("術後第 14 天", 14),

    ("術後第 30 天", 30),

    ("術後 30 天以上", 999),

]



def stage_tasks(stage_day: int):

    # 可依療程再細分模板；此處先做通用 demo

    if stage_day <= 3:

        return [

            ("冰敷 10–15 分鐘（每 2–3 小時一次）", True),

            ("加強保濕（至少 3 次）", True),

            ("避免熱敷、劇烈運動、烤箱/三溫暖", True),

            ("避免搓揉、去角質、酸類保養", True),

            ("外出防曬（遮蔽 + SPF）", True),

        ]

    if stage_day <= 14:

        return [

            ("加強保濕（至少 2–3 次）", True),

            ("避免摳痂/抓癢，讓其自然脫落", True),

            ("外出防曬（遮蔽 + SPF）", True),

            ("避免酸類/刺激性保養至穩定", True),

            ("每日溫和清潔（不過度清潔）", True),

        ]

    return [

        ("日常防曬（SPF + 遮蔽）", True),

        ("保濕維持（早晚）", True),

        ("避免過度去角質與刺激性療程", True),

        ("觀察是否有局部色素沉著並記錄", True),

    ]



def triage_from_symptoms(pain, heat, swelling, oozing, fever):

    """

    簡易分級：綠/黃/紅

    """

    # red flags

    if fever or oozing:

        return ("紅燈", "建議立即聯絡診所並安排回診；若合併劇痛、發燒或持續滲液，請立即就醫。", "pill-bad")

    if pain >= 7 or swelling >= 7:

        return ("紅燈", "疼痛/腫脹偏高，建議立即聯絡診所評估，並依醫師指示處理。", "pill-bad")

    if heat >= 6 or pain >= 5 or swelling >= 5:

        return ("黃燈", "症狀略高於一般預期，建議今日聯絡診所諮詢，並密切觀察是否加劇。", "pill-warn")

    return ("綠燈", "目前屬常見恢復反應，持續保濕、防曬與溫和照護即可。", "pill-good")



def explain_improvements(impr: dict, stage_day: int, q: dict):

    """

    把改善%轉成人話結論（核心：安心 + 可行動）

    """

    lines = []

    # Quality first

    if not q["ok"]:

        lines.append(f"本次照片品質評估：{q['score']} 分（{', '.join(q['tags']) if q['tags'] else '可再提升'}）。建議依拍攝指引重拍，以確保改善%具可比性。")

        return lines



    red = impr["redness"]

    spot = impr["spot"]

    wrinkle = impr["wrinkle"]



    if stage_day <= 3:

        lines.append("目前屬術後早期，泛紅與熱感波動屬常見現象；重點是穩定與舒緩。")

    elif stage_day <= 14:

        lines.append("進入代謝與修復期，膚況會逐步穩定；防曬與保濕會直接影響成效。")

    else:

        lines.append("膚況趨於穩定期，建議以維持型保養/療程延續效果。")



    # interpret a few metrics

    if red >= 12:

        lines.append(f"退紅改善明顯（+{red:.0f}%）：泛紅趨勢下降，代表恢復進度良好。")

    elif red <= -10:

        lines.append(f"退紅較前次偏弱（{red:.0f}%）：可能受光線或近期刺激影響，建議加強保濕與避免高溫刺激。")

    else:

        lines.append(f"退紅變化中（{red:.0f}%）：屬正常波動，請持續觀察趨勢。")



    if spot >= 8:

        lines.append(f"斑點指標提升（+{spot:.0f}%）：代謝啟動，後續 7–14 天通常更有感。")

    else:

        lines.append(f"斑點變化（{spot:.0f}%）：色素改善通常較慢，請以趨勢判讀。")



    if wrinkle >= 6:

        lines.append(f"紋理改善（+{wrinkle:.0f}%）：平滑度提升，與保濕與角質代謝相關。")



    return lines



# =========================================================

# 6) 主頁：登入/主程式

# =========================================================

def login_page():

    st.title("Beauty Tracker Login (專題 Demo)")

    st.caption("此版本聚焦核心價值：AI量化、改善%、成效報告感與術後照護體驗。")

    if st.button("登入測試帳號", type="primary"):

        st.session_state.logged_in = True

        st.session_state.user_key = DEMO_USER["phone"]

        st.rerun()



def main_app():

    user_key = st.session_state.user_key

    p = paths(user_key)

    engine = SkinEngine()



    history = load_history(user_key)



    with st.sidebar:

        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966334.png", width=80)

        st.title(DEMO_USER["name"])

        st.info(f"📋 療程：{DEMO_USER['treatment']}")

        st.caption(f"📅 療程日期：{DEMO_USER['op_date'].isoformat()}")

        st.markdown("---")

        if st.button("安全登出"):

            st.session_state.logged_in = False

            st.session_state.user_key = None

            st.rerun()



    st.markdown(f"## {APP_TITLE}")

    tab1, tab2, tab3 = st.tabs(["🩺 追蹤分析 (Live)", "📊 成效報告/歷史", "📅 預約回診"])



    # -----------------------------

    # Tab1: Live

    # -----------------------------

    with tab1:

        st.markdown("### 1) 選擇階段並上傳照片")

        with st.container(border=True):

            c1, c2, c3 = st.columns([2, 2, 2])



            with c1:

                stage_label = st.selectbox("術後階段", [s[0] for s in STAGES], index=0)

                stage_day = dict(STAGES)[stage_label]



            with c2:

                f_curr = st.file_uploader("上傳今日照片", type=["jpg", "jpeg", "png"], key="curr")



            with c3:

                if p["before_img"].exists():

                    st.success("✅ 術前圖已鎖定（此用戶）")

                    img_ref = load_image(p["before_img"])

                    if st.button("重新設定術前圖（慎用）"):

                        try:

                            p["before_img"].unlink(missing_ok=True)

                        except Exception:

                            pass

                        st.rerun()

                else:

                    f_ref = st.file_uploader("上傳術前圖（會鎖定）", type=["jpg", "jpeg", "png"], key="before")

                    img_ref = load_image(f_ref) if f_ref else None

                    if img_ref is not None and st.button("鎖定為術前圖", type="primary"):

                        save_rgb_image(img_ref, p["before_img"])

                        st.toast("✅ 術前圖已鎖定")

                        time.sleep(0.6)

                        st.rerun()



        if img_ref is None:

            st.info("請先鎖定術前圖，才能計算改善%。")

            return



        if f_curr is None:

            st.info("請上傳今日照片開始分析。")

            return



        img_curr = load_image(f_curr)

        with st.spinner("AI 運算中..."):

            aligned, _ = engine.align_faces(img_curr, img_ref)

            q = quality_check(aligned)

            res = engine.analyze(aligned)

            metrics = res["metrics"]

            avg = metrics_avg(metrics)



        # baseline metrics（從術前圖算一次）

        base_res = engine.analyze(img_ref)

        base_metrics = base_res["metrics"]



        # improvements (%)

        impr = {k: improvement_pct(metrics[k], base_metrics[k]) for k in metrics.keys()}



        # UI Layout

        colL, colR = st.columns([1.15, 1.0])



        with colL:

            st.markdown("### 2) 術前/目前影像對比與量化指標")



            image_comparison(img1=img_ref, img2=aligned, label1="術前", label2="目前", width=600, in_memory=True)



            # quality pill

            if q["score"] >= 80:

                pill_cls = "pill pill-good"

                q_text = "拍攝品質：優"

            elif q["score"] >= 60:

                pill_cls = "pill pill-warn"

                q_text = "拍攝品質：可"

            else:

                pill_cls = "pill pill-bad"

                q_text = "拍攝品質：需重拍"



            st.markdown(

                f'<div><span class="{pill_cls}">{q_text}</span>'

                f' <small>（亮度 {q["brightness"]} / 清晰度 {q["sharpness"]}）'

                f'{"｜問題：" + "、".join(q["tags"]) if q["tags"] else ""}</small></div>',

                unsafe_allow_html=True

            )



            st.markdown('<div class="hr"></div>', unsafe_allow_html=True)



            # top metrics cards

            k1, k2, k3 = st.columns(3)



            def metric_card(col, title, val, pct):

                sign = "+" if pct >= 0 else ""

                col.markdown(

                    f"""

                    <div style="text-align:center; padding:10px; border:1px solid #eee; border-radius:12px; background:white;">

                      <div class="metric-val">{val}</div>

                      <div class="metric-title">{title}</div>

                      <div class="metric-sub">改善 {sign}{pct:.0f}%（相對術前）</div>

                    </div>

                    """,

                    unsafe_allow_html=True

                )



            metric_card(k1, "退紅指數", metrics["redness"], impr["redness"])

            metric_card(k2, "斑點指數", metrics["spot"], impr["spot"])

            metric_card(k3, "綜合評分", avg, improvement_pct(avg, metrics_avg(base_metrics)))



            st.caption("註：改善%以術前作為基準；若拍攝品質不佳，改善%僅供參考。")



            st.markdown("### 3) 本次指標雷達圖")

            st.plotly_chart(plot_radar(metrics), use_container_width=True, key="live_radar")



        with colR:

            st.markdown("### 👩‍⚕️ 術後照護面板（讓人感覺被照顧）")



            # A) Symptoms & triage

            with st.container(border=True):

                st.markdown("#### A. 症狀回報（30 秒）")

                c1, c2 = st.columns(2)

                with c1:

                    pain = st.slider("疼痛程度", 0, 10, 2)

                    heat = st.slider("灼熱/熱感", 0, 10, 2)

                with c2:

                    swelling = st.slider("腫脹程度", 0, 10, 2)

                    oozing = st.checkbox("是否有滲液/水泡/明顯滲出？", value=False)

                    fever = st.checkbox("是否有發燒或全身不適？", value=False)



                level, msg, pill = triage_from_symptoms(pain, heat, swelling, oozing, fever)

                st.markdown(f'<div><span class="pill {pill}">風險分級：{level}</span></div>', unsafe_allow_html=True)

                st.write(msg)

                st.caption("此分級為追蹤提醒用途；若症狀快速加劇，請以專業醫療評估為準。")



            # B) Tasks checklist

            with st.container(border=True):

                st.markdown("#### B. 今日照護任務清單")

                tasks = stage_tasks(stage_day)

                # session key for tasks

                t_key = f"tasks_{stage_day}"

                if t_key not in st.session_state:

                    st.session_state[t_key] = {t[0]: False for t in tasks}



                done = 0

                for t, default in tasks:

                    st.session_state[t_key][t] = st.checkbox(t, value=st.session_state[t_key].get(t, False))

                    if st.session_state[t_key][t]:

                        done += 1



                total = max(1, len(tasks))

                st.progress(done / total)

                st.write(f"今日完成度：{int(done/total*100)}%")



            # C) Nurse SOP conclusion (based on stage + improvements + quality)

            with st.container(border=True):

                st.markdown("#### C. AI 護理師結論（可理解、可行動）")



                lines = explain_improvements(impr, stage_day, q)

                for ln in lines:

                    st.write(f"- {ln}")



                st.markdown("**下一步建議**")

                if stage_day <= 3:

                    st.write("- 今日重點：舒緩（冰敷/保濕）與避免刺激。")

                    st.write("- 若疼痛、腫脹快速上升或出現滲液/發燒，請立即聯絡診所。")

                elif stage_day <= 14:

                    st.write("- 今日重點：保濕與防曬，避免摳痂與刺激性保養。")

                    st.write("- 若泛紅持續加劇或局部熱痛明顯，建議回診評估。")

                else:

                    st.write("- 今日重點：防曬與穩定保養，維持療程效果。")



            # Save record

            st.markdown("### 4) 存入病歷（含照片與改善%）")

            can_save = True

            if not q["ok"]:

                st.warning("照片品質不足，建議重拍後再存檔（避免改善%失真）。")

                can_save = False



            if st.button("💾 存入病歷", type="primary", use_container_width=True, disabled=not can_save):

                rec_id = str(uuid.uuid4())

                img_path = p["records_dir"] / f"{rec_id}.jpg"

                save_rgb_image(aligned, img_path)



                record = {

                    "id": rec_id,

                    "stage_label": stage_label,

                    "stage_day": stage_day,

                    "record_date": date.today().isoformat(),

                    "metrics": metrics,

                    "baseline_metrics": base_metrics,

                    "improvement_pct": {k: round(float(impr[k]), 2) for k in impr.keys()},

                    "quality": q,

                    "avg": avg,

                    "img_path": str(img_path),

                    "symptoms": {"pain": pain, "heat": heat, "swelling": swelling, "oozing": oozing, "fever": fever, "triage": level},

                    "tasks_done_pct": int(done / total * 100),

                }

                history.append(record)

                save_history(user_key, history)

                st.toast("✅ 已存入病歷（本次改善%已記錄）")

                time.sleep(0.8)

                st.rerun()



    # -----------------------------

    # Tab2: History / Report

    # -----------------------------

    with tab2:

        history = load_history(user_key)

        if not history:

            st.info("尚無歷史數據。建議先在 Live 頁存入一筆病歷。")

        else:

            st.markdown("### 📈 成效趨勢（自動排序）")

            st.plotly_chart(plot_trend(history), use_container_width=True, key="history_trend")



            st.markdown("### 🗂️ 歷史紀錄（含改善% 與品質）")

            # newest first

            hist_sorted = sorted(history, key=lambda r: (int(r.get("stage_day", 999)), r.get("record_date", "")), reverse=True)



            for i, rec in enumerate(hist_sorted):

                with st.container(border=True):

                    top = st.columns([1.2, 2.0])

                    with top[0]:

                        if rec.get("img_path") and os.path.exists(rec["img_path"]):

                            st.image(rec["img_path"], caption=f"{rec['stage_label']}｜{rec['record_date']}")

                        else:

                            st.info("照片檔案不存在")

                    with top[1]:

                        q = rec.get("quality", {})

                        q_score = int(q.get("score", 0))

                        if q_score >= 80:

                            pill_cls = "pill pill-good"

                            q_text = "品質：優"

                        elif q_score >= 60:

                            pill_cls = "pill pill-warn"

                            q_text = "品質：可"

                        else:

                            pill_cls = "pill pill-bad"

                            q_text = "品質：弱"



                        st.markdown(f"**{rec['stage_label']}**  <span class='{pill_cls}'>{q_text} {q_score}</span>", unsafe_allow_html=True)

                        st.write(f"- 綜合評分：{rec.get('avg', 0)}")

                        imp = rec.get("improvement_pct", {})

                        st.write(f"- 退紅改善：{imp.get('redness', 0)}%｜斑點改善：{imp.get('spot', 0)}%｜紋理改善：{imp.get('wrinkle', 0)}%")

                        sym = rec.get("symptoms", {})

                        if sym:

                            st.write(f"- 風險分級：{sym.get('triage','-')}｜疼痛 {sym.get('pain','-')}｜熱感 {sym.get('heat','-')}｜腫脹 {sym.get('swelling','-')}")

                        st.write(f"- 今日照護完成度：{rec.get('tasks_done_pct', 0)}%")



                        st.plotly_chart(plot_radar(rec["metrics"]), use_container_width=True, key=f"history_radar_{i}")



            st.caption("你可以在此頁截圖當作『成效報告展示』；若要做成 PDF，可再加一個 report 生成按鈕。")



    # -----------------------------

    # Tab3: Appointment

    # -----------------------------

    with tab3:

        st.subheader("📅 預約回診（Demo）")

        appt_date = st.date_input("日期", value=date.today() + timedelta(days=7))

        appt_note = st.text_input("備註（可選）", value="術後追蹤回診")

        if st.button("確認預約", type="primary"):

            st.success(f"已送出預約需求：{appt_date.isoformat()}（{appt_note}）")

            st.caption("專題版可先示範流程；正式版再串診所預約系統或訊息推播。")



# =========================================================

# 7) 執行

# =========================================================

if __name__ == "__main__":

    if st.session_state.logged_in:

        main_app()

    else:

        login_page()

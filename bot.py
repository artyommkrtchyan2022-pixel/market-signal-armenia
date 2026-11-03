#!/usr/bin/env python3
import json, logging, os, sys, threading, time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger("signal-bot")
YEREVAN = ZoneInfo("Asia/Yerevan")
PAIRS = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"}
DIGITS = {"EUR/USD": 5, "GBP/USD": 5, "USD/JPY": 3}
TIMEFRAMES = {
    "1m": {"minutes": 1, "period": "5d", "stale_min": 10, "label": "1 րոպե"},
    "5m": {"minutes": 5, "period": "5d", "stale_min": 30, "label": "5 րոպե"},
}
EMA_FAST, EMA_SLOW = 9, 21
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_PERIOD = 14
MIN_CANDLES = 60
MIN_EMA_GAP = 0.00002
COOLDOWN_CANDLES = 3
ALERT_AFTER_FAILED_CYCLES = 10
ALERT_REPEAT_SECONDS = 3 * 3600
BOT_TOKEN = None
CHAT_ID = None
SEND_WAIT = True
STATE_FILE = "state.json"
WARNING_HY = ("⚠️ <b>Զգուշացում՝</b> Yahoo Finance-ի գները կարող են ուշանալ և տարբերվել "
              "Pocket Option-ի գներից։ Սա ֆինանսական խորհուրդ չէ, առևտուրը բարձր ռիսկային է։ "
              "Բոտը գործարքներ չի կատարում։")
SIGNAL_STYLE = {
    "BUY": ("🟢", "ԳՆԵԼ", "BUY", " · CALL ⬆️"),
    "SELL": ("🔴", "ՎԱՃԱՌԵԼ", "SELL", " · PUT ⬇️"),
    "WAIT": ("⚪️", "ՍՊԱՍԵԼ", "WAIT", ""),
}

class StaleDataError(Exception):
    pass

def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return {}

def save_state(state):
    try:
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        log.warning("Could not save state: %s", type(e).__name__)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK - signal bot is running")
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        pass

def start_health_server():
    port = os.environ.get("PORT")
    if port:
        try:
            server = HTTPServer(("0.0.0.0", int(port)), HealthHandler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except Exception as e:
            log.warning("Health server failed: %s", type(e).__name__)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    for attempt in range(3):
        try:
            r = requests.post(url, data=payload, timeout=15)
            if r.status_code == 200:
                return True
            if r.status_code == 429:
                try:
                    wait = int(r.json().get("parameters", {}).get("retry_after", 5))
                except Exception:
                    wait = 5
                time.sleep(min(wait, 30))
                continue
            log.error("Telegram error %s: %s", r.status_code, r.text[:200])
            if 400 <= r.status_code < 500:
                return False
        except requests.RequestException as e:
            log.warning("Telegram network error: %s", type(e).__name__)
        time.sleep(2 * (attempt + 1))
    return False

def fetch_candles(symbol, tf_key):
    import yfinance as yf
    cfg = TIMEFRAMES[tf_key]
    df = yf.Ticker(symbol).history(period=cfg["period"], interval=tf_key,
                                   auto_adjust=False, actions=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no data")
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    delta = pd.Timedelta(minutes=cfg["minutes"])
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + delta <= now]
    if len(df) < MIN_CANDLES:
        raise RuntimeError(f"not enough candles ({len(df)})")
    age_min = (now - (df.index[-1] + delta)).total_seconds() / 60
    if age_min > cfg["stale_min"]:
        raise StaleDataError(f"last candle is {age_min:.0f} min old")
    return df.tail(300)

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    out = 100 - 100 / (1 + rs)
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    return out

def analyze(df):
    close = df["Close"]
    ef, es = ema(close, EMA_FAST), ema(close, EMA_SLOW)
    ml = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    ms = ema(ml, MACD_SIGNAL)
    hist, rs = ml - ms, rsi(close, RSI_PERIOD)
    v = {"price": float(close.iloc[-1]), "ema_fast": float(ef.iloc[-1]),
         "ema_slow": float(es.iloc[-1]), "macd": float(ml.iloc[-1]),
         "macd_signal": float(ms.iloc[-1]), "hist": float(hist.iloc[-1]),
         "rsi": float(rs.iloc[-1])}
    if any(pd.isna(x) for x in v.values()):
        raise ValueError("indicators not ready")
    gap_ok = abs(v["ema_fast"] - v["ema_slow"]) / v["price"] >= MIN_EMA_GAP
    up = v["ema_fast"] > v["ema_slow"] and v["macd"] > v["macd_signal"] and 50 < v["rsi"] < 70 and gap_ok
    down = v["ema_fast"] < v["ema_slow"] and v["macd"] < v["macd_signal"] and 30 < v["rsi"] < 50 and gap_ok
    v["signal"] = "BUY" if up else "SELL" if down else "WAIT"
    return v

def build_message(pair, tf_key, res, candle_start):
    cfg, d = TIMEFRAMES[tf_key], DIGITS[pair]
    emoji, hy, en, extra = SIGNAL_STYLE[res["signal"]]
    close_time = (candle_start + pd.Timedelta(minutes=cfg["minutes"])).tz_convert(YEREVAN)
    now_local = datetime.now(YEREVAN)
    trend = "EMA 9-ը EMA 21-ից բարձր է (աճող միտում)" if res["ema_fast"] > res["ema_slow"] else "EMA 9-ը EMA 21-ից ցածր է (նվազող միտում)"
    macd_txt = "MACD գիծը ազդանշանային գծից բարձր է" if res["macd"] > res["macd_signal"] else "MACD գիծը ազդանշանային գծից ցածր է"
    rsi_txt = f"RSI-ը {res['rsi']:.1f} է"
    if res["rsi"] >= 70: rsi_txt += " (գերգնված գոտի)"
    elif res["rsi"] <= 30: rsi_txt += " (գերվաճառված գոտի)"
    lines = [f"{emoji} <b>{hy} ({en})</b>{extra}", "", f"💱 Զույգ՝ <b>{pair}</b>",
             f"⏱ Ժամանակահատված՝ <b>{cfg['label']}</b>", f"💰 Գին՝ <b>{res['price']:.{d}f}</b>",
             f"🕒 Մոմի փակում՝ {close_time:%H:%M} (Երևանի ժամանակով)",
             f"📨 Ազդանշանի ժամը՝ {now_local:%H:%M:%S}"]
    if res["signal"] in ("BUY", "SELL"):
        lines.append(f"⌛ Առաջարկվող ժամկետ՝ {cfg['label']}")
    lines += ["", "📊 <b>Ցուցանիշներ՝</b>", f"• RSI (14)՝ {res['rsi']:.1f}",
              f"• EMA 9՝ {res['ema_fast']:.{d}f} | EMA 21՝ {res['ema_slow']:.{d}f}",
              f"• MACD՝ {res['macd']:.{d+1}f} | Ազդանշան՝ {res['macd_signal']:.{d+1}f}",
              f"• Հիստոգրամ՝ {res['hist']:.{d+1}f}", "", "🧠 <b>Հիմնավորում՝</b>",
              f"• {trend}", f"• {macd_txt}", f"• {rsi_txt}"]
    if res["signal"] == "WAIT":
        lines.append("• Ցուցանիշները միմյանց չեն հաստատում, հստակ մուտք չկա։")
    lines += ["", WARNING_HY]
    return "\n".join(lines)

def process(state, pair, tf_key):
    key = f"{pair}|{tf_key}"
    cfg = TIMEFRAMES[tf_key]
    df = fetch_candles(PAIRS[pair], tf_key)
    candle_start = df.index[-1]
    candle_id = candle_start.isoformat()
    st = state.get(key, {})
    if st.get("candle") == candle_id:
        return
    res = analyze(df)
    signal, last, now_ts = res["signal"], st.get("signal"), time.time()
    should_send = False
    if signal != last:
        if signal in ("BUY", "SELL"):
            cooldown = COOLDOWN_CANDLES * cfg["minutes"] * 60
            recent_same = st.get("dir") == signal and now_ts - st.get("dir_time", 0) < cooldown
            should_send = not recent_same
        else:
            should_send = SEND_WAIT and last in ("BUY", "SELL")
    if should_send:
        if not send_telegram(build_message(pair, tf_key, res, candle_start)):
            log.warning("Signal %s %s %s not delivered, will retry", pair, tf_key, signal)
            return
        log.info("Sent %s %s %s", signal, pair, tf_key)
        if signal in ("BUY", "SELL"):
            st["dir"], st["dir_time"] = signal, now_ts
    st["signal"], st["candle"] = signal, candle_id
    state[key] = st
    save_state(state)

def run_cycle(state, tf_keys):
    ok = fails = 0
    for tf_key in tf_keys:
        for pair in PAIRS:
            try:
                process(state, pair, tf_key)
                ok += 1
            except StaleDataError as e:
                log.info("%s %s skipped: %s", pair, tf_key, e)
                ok += 1
            except Exception as e:
                fails += 1
                log.warning("%s %s error: %s: %s", pair, tf_key, type(e).__name__, str(e)[:150])
            time.sleep(1)
    return ok, fails

def sleep_until_next_minute(offset=5):
    now = time.time()
    target = (now // 60 + 1) * 60 + offset
    time.sleep(max(1, target - now))

def main():
    global BOT_TOKEN, CHAT_ID, SEND_WAIT, STATE_FILE
    BOT_TOKEN = os.environ.get("BOT_TOKEN", "").strip()
    CHAT_ID = os.environ.get("CHAT_ID", "").strip()
    SEND_WAIT = os.environ.get("SEND_WAIT", "true").strip().lower() in ("1", "true", "yes", "on")
    STATE_FILE = os.environ.get("STATE_FILE", "state.json")
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(message)s")
    if not BOT_TOKEN or not CHAT_ID:
        log.error("BOT_TOKEN and CHAT_ID environment variables are required.")
        sys.exit(1)
    start_health_server()
    state = load_state()
    if not send_telegram("🤖 <b>Բոտը գործարկվեց</b>\nՎերլուծվող զույգեր՝ EUR/USD, GBP/USD, USD/JPY\n"
                         "Ժամանակահատվածներ՝ 1 րոպե և 5 րոպե\nՑուցանիշներ՝ RSI, EMA 9, EMA 21, MACD\n\n" + WARNING_HY):
        log.error("Startup message failed. Check BOT_TOKEN / CHAT_ID and press START in Telegram.")
    consecutive_bad, last_alert, first = 0, 0.0, True
    while True:
        try:
            if first:
                tf_keys, first = ["1m", "5m"], False
            else:
                sleep_until_next_minute()
                minute = datetime.now(timezone.utc).minute
                tf_keys = ["1m"] + (["5m"] if minute % 5 == 0 else [])
            ok, fails = run_cycle(state, tf_keys)
            consecutive_bad = consecutive_bad + 1 if ok == 0 and fails > 0 else 0
            if consecutive_bad >= ALERT_AFTER_FAILED_CYCLES and time.time() - last_alert > ALERT_REPEAT_SECONDS:
                send_telegram("🛠 <b>Սխալ՝</b> Բոտը չի կարողանում ստանալ տվյալներ Yahoo Finance-ից։ Փորձերը շարունակվում են։")
                last_alert = time.time()
        except KeyboardInterrupt:
            log.info("Stopped by user")
            break
        except Exception as e:
            log.error("Unexpected loop error: %s: %s", type(e).__name__, str(e)[:150])
            time.sleep(10)

if __name__ == "__main__":
    main()

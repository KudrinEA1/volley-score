"""Сервер счёта. Только стандартная библиотека + OpenCV/Tesseract (через recognizer.py).

Запуск:  python server.py          (порт берётся из переменной PORT, по умолчанию 8000)

Адреса:
  GET  /            страница для телефона B (камера, рамка, ручные поправки)
  GET  /overlay     прозрачная графика со счётом (для наложения на трансляцию)
  GET  /score       текущий счёт в JSON
  GET  /latest.jpg  последний присланный кадр (для проверки)
  GET  /frames.zip  архив сохранённых кадров для обучения (?key=КЛЮЧ&n=60&flag=all|LOST|chg|smp)
  POST /frame       JPEG-кадр (?swap=1, если наши справа; ?mode=full - весь кадр, табло ищется само; ?mode=crop - кадр вокруг табло;
                    ?air=1 - страница в эфире; ?cid= - идентификатор страницы)
  POST /air         {"cid": "...", "on": true|false}  взять эфир этой страницей / снять
  POST /source      {"source": "camera|manual"}       кто ведёт счёт: камера или только кнопки
  POST /undo        отменить последнее ручное действие
  POST /reset_set   счёт партии в 0:0 (партии остаются)
  POST /close_set   закрыть партию в статистике вручную
  POST /showstats   {"on": true|false}  таблица статистики на плашке (сама исчезает через 60 с)
  POST /settings    {"serve": true|false}  показывать ли точку подачи
  GET  /control     пульт: та же страница без камеры, с любого телефона
  POST /side        {"swap": bool}  где наши на табло (только указание)
  POST /side_answer {"us_left": bool}  ответ после конца партии: команды переезжают вместе со счётом
  POST /match/start {"us","them"}  /match/end  /set_end  (конец партии вручную)
  POST /pit/start  /pit/draft {"seq":[[команда 0/1, порядок неизвестен 0/1]...], "sets":[наши, соперник]}  /pit/done  /pit/cancel
  GET  /stats       страница статистики матча;  GET /stats.json - то же данными
  GET  /version     номер сборки
  GET  /backup      полная копия состояния (её хранят телефоны)
  POST /restore     вернуть копию на сервер, запустившийся пустым (после перезапуска хостинга)
  POST /adjust      {"who": "us|them", "field": "points|sets", "delta": 1|-1}
  POST /reset       обнулить счёт
  POST /names       {"us": "...", "them": "..."}
"""
import io
import json
import os
import zipfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

import recognizer as recog
from recognizer import find_board, read_board, valid_points, valid_sets
from score import MatchStats, ScoreState

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
os.makedirs(DATA, exist_ok=True)

STATE = ScoreState(set_points=25, last_set_points=15, sets_to_win=3,
                   path=os.path.join(DATA, "state.json"))
FRAME_LOCK = threading.Lock()          # кадры обрабатываются по одному
MAX_BODY = 5_000_000

# История кадров для обучения распознавания (раньше хранился только последний кадр и затирался следующим).
# Архив: /frames.zip?key=КЛЮЧ  (ключ защищает от чужих: в кадрах бывают люди в зале)
FRAMES_KEY = os.environ.get("FRAMES_KEY", "volley-7QmK3xP2")
FRAMES_DIR = os.path.join(DATA, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)
MAX_FRAMES = 400
MAX_LOST_CROPS = 100                   # нечитаемые кадры-вырезки: не больше стольких, чтобы они не вытесняли всё остальное
BUILD = 17                   # номер сборки: страница сверяет его со своим
FLOG = {"pts": None, "saved": 0.0}

# Эфир. Счёт на плашке меняет только ГЛАВНАЯ камера: страница, на которой последней нажали «В эфир».
# Остальные страницы работают в режиме «Проверка»: табло ищут и читают, но счёт не трогают.
# source: "camera" - счёт ведёт камера; "manual" - только кнопки, камера на счёт не влияет.
AIR_PATH = os.path.join(DATA, "air.json")
AIR = {"cid": None, "last": 0.0, "source": "camera"}
try:
    with open(AIR_PATH, encoding="utf-8") as _f:
        _a = json.load(_f)
        AIR["source"] = _a.get("source", "camera")
        AIR["cid"] = _a.get("cid")                     # главная камера помнится и после перезапуска сервера
except (OSError, ValueError):
    _a = {}

# ---------- копия состояния на телефонах ----------
# На бесплатном хостинге папка data/ стирается при каждом перезапуске. Поэтому страницы (телефон B и пульты)
# каждые несколько секунд забирают копию /backup и хранят её у себя. Сервер запустился пустым (fresh) -
# страница присылает копию в /restore, и счёт продолжается с того же места.
BOOT = os.urandom(6).hex()                             # метка запуска: сменилась - значит, сервер перезапускался
RESTORE = {"fresh": not os.path.exists(os.path.join(DATA, "state.json")),   # пустой запуск: копию можно принять
           "saved": 0.0, "at": 0.0}
RESTORED_SHOW_SEC = 120
# настройки с телефона B: показывать ли точку подачи; есть ли на табло счёт партий (если нет - камера партии не читает,
# партии считает сервер сам: 25 очков с разницей 2, затем табло сброшено в 0:0)
SETTINGS = {"serve": bool(_a.get("serve", True)), "board_sets": bool(_a.get("board_sets", False))}
STATS = MatchStats(path=os.path.join(DATA, "stats.json"))
SHOW = {"until": 0.0}                    # «Статистика в эфир»: до какого времени показывать таблицу на плашке
SHOW_SEC = 60
SUMMARY_SEC = 10                         # сколько секунд плашка показывает итог закончившейся партии
SERVE = {"prev": None, "who": None, "hist": []}
UNDO = []                               # снимки счёта перед ручными действиями (для «Отменить последнее»)
UNDO_MAX = 30

# ---------- режим матча ----------
# Матч задаёт контекст: формат, названия, основной трек счёта. Камера только продолжает трек.
# Питстоп: камера не влияет на счёт, статистика застывает, плашка серая «Счёт уточняется»; счёт правят вручную
# (черновик ленты розыгрышей), «Готово» делает его правдой. Автопитстоп: камера долго не видит логичного счёта.
MATCH_PATH = os.path.join(DATA, "match.json")
MATCHES_DIR = os.path.join(DATA, "matches")
os.makedirs(MATCHES_DIR, exist_ok=True)
AUTOPIT_SEC = 30                         # столько секунд без логичного счёта от камеры - и включается питстоп сам
PIT_GRACE_SEC = 30                       # после «Готово» автопитстоп столько секунд не срабатывает
MIRROR_SEC = 8                           # камера столько секунд видит зеркальный счёт - предупреждение «стороны перепутаны»
MATCH = {"active": False, "started": None, "pit": None, "grace": 0.0, "prompt": None, "prev": None,
         "mid_swap_set": None, "mirror_since": None}
try:
    with open(MATCH_PATH, encoding="utf-8") as _f:
        MATCH.update(json.load(_f))
except (OSError, ValueError):
    pass
STATE.strict = MATCH["active"]
if not SETTINGS["board_sets"]:
    STATE.sets_ready = False                # партий на табло нет: сервер считает их сам
AUTOPIT_EXIT_SEC = 10                    # автопитстоп снимается сам, если камера столько секунд видит один и тот же правдоподобный счёт
AUTOPIT_EXIT_FRAMES = 3
HEALED_SHOW_SEC = 120


def save_match():
    try:
        with open(MATCH_PATH, "w", encoding="utf-8") as f:
            json.dump(MATCH, f, ensure_ascii=False)
    except OSError:
        pass


def _team_pts():
    sc = STATE.snapshot()
    return sc["us"]["points"], sc["them"]["points"], sc["us"]["sets"], sc["them"]["sets"]


def match_start():
    STATE.reset()
    STATS.reset_match()
    SERVE.update(prev=(0, 0), who=None, hist=[])
    with STATE.lock:
        STATE.points_ready = True                # матч начинается с 0:0: это уже подтверждённый счёт
        STATE.save()
    MATCH.update(active=True, started=time.time(), pit=None, grace=time.time() + PIT_GRACE_SEC, prompt=None,
                 prev=[0, 0], mid_swap_set=None, mirror_since=None)
    STATE.strict = True
    STATE.last_ok = time.time()
    LASTS.clear()
    save_match()


def match_end():
    us, th, us_s, th_s = _team_pts()
    rec = {"started": MATCH["started"], "ended": time.time(), "names": dict(STATE.names),
           "sets": [us_s, th_s], "table": STATS.table()}
    name = time.strftime("%Y%m%d-%H%M", time.localtime(MATCH["started"] or time.time())) + ".json"
    try:
        with open(os.path.join(MATCHES_DIR, name), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)
    except OSError:
        pass
    MATCH.update(active=False, pit=None, prompt=None)
    STATE.strict = False
    save_match()
    return name


def pit_start(auto=False):
    if MATCH["pit"]:
        return
    us, th, us_s, th_s = _team_pts()
    STATS.sync(us_s + th_s)
    seq = [list(x) for x in (STATS.dump()["sets"][-1]["seq"] if STATS.sets else [])]
    # черновик сразу равен счёту на плашке: если статистика отстала - добавляем очки «порядок неизвестен», если убежала - убираем
    for team, need in ((0, us), (1, th)):
        have = sum(1 for t, _ in seq if t == team)
        for _ in range(need - have):
            seq.append([team, 1])
        for _ in range(have - need):
            for k in range(len(seq) - 1, -1, -1):
                if seq[k][0] == team:
                    del seq[k]
                    break
    MATCH["pit"] = {"auto": auto, "since": time.time(),
                    "draft": {"seq": seq, "sets": [us_s, th_s]},
                    "before": {"points": [us, th], "sets": [us_s, th_s], "seq": [list(x) for x in seq]}}
    save_match()


def pit_done():
    p = MATCH["pit"]
    if not p:
        return
    remember()
    seq = [[int(t), int(q)] for t, q in p["draft"]["seq"] if t in (0, 1)]
    us = sum(1 for t, _ in seq if t == 0)
    th = len(seq) - us
    sets = p["draft"]["sets"]
    STATE.set_team_values(us, th, sets[0], sets[1])
    STATS.sync(sets[0] + sets[1])                 # партии изменились в питстопе: прошлая закрывается, лента - уже новой партии
    with STATS.lock:
        STATS.sets[-1]["seq"] = seq
        STATS.cur, STATS.cand = [us, th], None
        STATS.save()
    SERVE.update(prev=(us, th), who=None, hist=[])
    MATCH.update(pit=None, grace=time.time() + PIT_GRACE_SEC, prev=[us, th])
    LASTS.clear()
    save_match()


def pit_heal(sn):
    """Автопитстоп снимается сам: камера AUTOPIT_EXIT_SEC секунд подряд видит один и тот же правдоподобный счёт
    (не меньше, чем был при входе в питстоп). Если в питстопе правили черновик вручную - ждём «Готово» от человека."""
    p = MATCH["pit"]
    if not p or not p.get("auto") or p.get("touched"):
        return
    now = time.time()
    b = p["before"]["points"]
    c = [sn["us"], sn["them"]] if sn else None
    if not c or None in c or c[0] < b[0] or c[1] < b[1] or not valid_points(c[0], c[1], STATE._target()):
        p.pop("heal", None)                       # ничего понятного не видно: ждём дальше
        return
    h = p.get("heal")
    if not h or h["score"] != c:
        p["heal"] = {"score": c, "since": now, "n": 1}
        return
    h["n"] += 1
    if now - h["since"] < AUTOPIT_EXIT_SEC or h["n"] < AUTOPIT_EXIT_FRAMES:
        return
    us_s, th_s = p["before"]["sets"]
    STATE.set_team_values(c[0], c[1], us_s, th_s)
    MATCH.update(pit=None, grace=now + PIT_GRACE_SEC, prev=list(c), healed={"at": now, "score": c})
    STATS.observe(tuple(c), STATE._target(), immediate=True)    # очки, пропущенные в питстопе, - «порядок неизвестен»
    SERVE.update(prev=tuple(c), who=None, hist=[])
    save_match()


def pit_cancel():
    MATCH.update(pit=None, grace=time.time() + PIT_GRACE_SEC)
    STATE.last_ok = time.time()
    save_match()


def set_side_prompt(n, mid=False):
    MATCH["prompt"] = {"set": n, "mid": mid, "since": time.time()}
    save_match()


def track():
    """Следит за ходом матча: конец партии -> смена сторон и вопрос «где наши»; пятая партия на 8 очках;
    автопитстоп; зеркальный счёт. Вызывается при каждом обращении к счёту."""
    if not MATCH["active"]:
        return
    now = time.time()
    us, th, us_s, th_s = _team_pts()
    prev = MATCH.get("prev") or [0, 0]
    target = STATE._target()
    # новая партия: был законно законченный счёт, стало 0:0 -> по умолчанию стороны меняются (на вопрос можно ответить иначе)
    if (us, th) == (0, 0) and tuple(prev) != (0, 0) and max(prev) >= 20:
        STATE.flip_sides()
        set_side_prompt(us_s + th_s + 1)
        MATCH["mid_swap_set"] = None
    # пятая (решающая) партия: смена сторон, когда у кого-то 8 очков
    n_set = us_s + th_s + 1
    if target == STATE.last_set_points and max(us, th) >= 8 and MATCH.get("mid_swap_set") != n_set:
        MATCH["mid_swap_set"] = n_set
        STATE.flip_sides()
        set_side_prompt(n_set, mid=True)
    if [us, th] != prev:
        MATCH["prev"] = [us, th]
        save_match()
    # вопрос о сторонах убираем, когда партия ушла вперёд
    pr = MATCH.get("prompt")
    if pr and (us + th >= 6 + (8 if pr.get("mid") else 0) or now - pr["since"] > 300):
        MATCH["prompt"] = None
        save_match()
    # автопитстоп
    if (not MATCH["pit"] and AIR["cid"] and AIR["source"] == "camera" and now > MATCH["grace"]
            and now - STATE.last_ok > AUTOPIT_SEC):
        pit_start(auto=True)


def side_answer(us_left):
    """Ответ на вопрос «где наши на табло»: если не совпадает с текущим - команды меняются местами вместе со счётом."""
    if bool(us_left) == (not STATE.swap):
        pass
    else:
        STATE.flip_sides()
    MATCH["prompt"] = None
    save_match()


def save_air():
    try:
        with open(AIR_PATH, "w", encoding="utf-8") as f:
            json.dump({"source": AIR["source"], "serve": SETTINGS["serve"], "board_sets": SETTINGS["board_sets"],
                       "cid": AIR["cid"]}, f)
    except OSError:
        pass


def backup():
    """Полная копия состояния для хранения на телефонах (/backup)."""
    state = None
    try:
        with open(STATE.path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        pass
    matches = {}
    for name in sorted(x for x in os.listdir(MATCHES_DIR) if x.endswith(".json"))[-10:]:
        try:
            with open(os.path.join(MATCHES_DIR, name), encoding="utf-8") as f:
                matches[name] = json.load(f)
        except (OSError, ValueError):
            pass
    return {"boot": BOOT, "build": BUILD, "saved": time.time(), "fresh": RESTORE["fresh"],
            "restored": bool(RESTORE["at"]), "state": state, "stats": STATS.dump(),
            "match": dict(MATCH), "air": {"source": AIR["source"], "serve": SETTINGS["serve"],
                                          "board_sets": SETTINGS["board_sets"], "cid": AIR["cid"]},
            "matches": matches}


def restore(b):
    """Принять копию с телефона. Только если сервер запустился пустым и после запуска счёт никто не трогал;
    из нескольких копий (несколько телефонов) остаётся самая свежая."""
    if not RESTORE["fresh"]:
        return False, "сервер не пустой: копия не нужна"
    if not isinstance(b, dict) or b.get("boot") == BOOT or not isinstance(b.get("state"), dict):
        return False, "копия не подходит"
    saved = float(b.get("saved") or 0)
    now = time.time()
    if RESTORE["at"] and now - RESTORE["at"] > 60:
        return False, "восстановление уже было, счёт ушёл вперёд"
    if saved <= RESTORE["saved"]:
        return False, "уже восстановлено из более свежей копии"
    with open(STATE.path, "w", encoding="utf-8") as f:
        json.dump(b["state"], f, ensure_ascii=False)
    with STATE.lock:
        STATE.load()
        STATE._pc, STATE._pn, STATE._sc, STATE._sn, STATE._fin_n = None, 0, None, 0, 0
        STATE.status = "счёт восстановлен после перезапуска сервера"
        STATE.updated = now
        STATE.last_ok = now
    if isinstance(b.get("stats"), dict) and b["stats"].get("sets"):
        STATS.restore(b["stats"])
    if isinstance(b.get("match"), dict):
        MATCH.update(b["match"])
        MATCH["grace"] = now + PIT_GRACE_SEC               # сразу после восстановления автопитстоп не включаем
        MATCH["mirror_since"] = None
        save_match()
    STATE.strict = MATCH["active"]
    a = b.get("air") or {}
    if a.get("source") in ("camera", "manual"):
        AIR["source"] = a["source"]
    if "serve" in a:
        SETTINGS["serve"] = bool(a["serve"])
    if "board_sets" in a:
        SETTINGS["board_sets"] = bool(a["board_sets"])
    if not SETTINGS["board_sets"]:
        STATE.sets_ready = False
    if a.get("cid") and not AIR["cid"]:
        AIR["cid"], AIR["last"] = str(a["cid"])[:12], now
    save_air()
    for name, rec in (b.get("matches") or {}).items():
        name = os.path.basename(str(name))
        p = os.path.join(MATCHES_DIR, name)
        if name.endswith(".json") and not os.path.exists(p):
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False)
            except OSError:
                pass
    us, th, _, _ = _team_pts()
    SERVE.update(prev=(us, th), who=None, hist=[])
    LASTS.clear()
    RESTORE.update(saved=saved, at=now)
    return True, "счёт восстановлен"


def _serve(sc):
    """Подаёт команда, взявшая последнее очко. Неясно (начало партии, скачок, обе изменились) - точки нет."""
    p = SERVE["prev"]
    SERVE["prev"] = sc
    if p is None or sc == p:
        return
    d = (sc[0] - p[0], sc[1] - p[1])
    if d in ((1, 0), (0, 1)):
        SERVE["hist"].append(SERVE["who"])
        del SERVE["hist"][:-50]
        SERVE["who"] = "us" if d == (1, 0) else "them"
    elif d in ((-1, 0), (0, -1)):                       # судья исправил: возвращаем, как было
        SERVE["who"] = SERVE["hist"].pop() if SERVE["hist"] else None
    else:
        SERVE["who"] = None
        SERVE["hist"].clear()


def payload():
    """Счёт + состояние эфира + всё для плашки. Это же отдаёт /score."""
    now = time.time()
    track()
    s = STATE.snapshot()
    us, th = s["us"]["points"], s["them"]["points"]
    target = STATE._target()
    pit = MATCH["pit"]
    if not pit:                                         # в питстопе статистика застывает
        STATS.sync(s["us"]["sets"] + s["them"]["sets"])   # номер партии в статистике = сыграно партий + 1
        STATS.observe((us, th), target)
    _serve((us, th))
    age = round(now - AIR["last"], 1) if AIR["cid"] and AIR["last"] else None
    s["air"] = {"on": bool(AIR["cid"]), "source": AIR["source"], "frame_age": age}
    s["build"] = BUILD
    s["boot"] = BOOT
    s["fresh"] = RESTORE["fresh"]
    s["settings"] = dict(SETTINGS)
    s["serve"] = SERVE["who"] if SETTINGS["serve"] else None
    s["ball"] = None                                       # сетбол / матчбол
    for who, mine, other, sets in (("us", us, th, s["us"]["sets"]), ("them", th, us, s["them"]["sets"])):
        if mine >= target - 1 and mine > other and not STATE.is_finished((us, th)):
            s["ball"] = {"who": who, "kind": "match" if sets == STATE.sets_to_win - 1 else "set"}
    team, n = STATS.streak()
    s["streak"] = {"who": "us" if team == 0 else "them", "n": n} if n >= 3 else None
    n_closed, last = STATS.last_closed()
    s["summary"] = ({"n": n_closed, "final": last["final"], "seq": last["seq"], "end": last["end"]}
                    if last and now - last["end"] < SUMMARY_SEC else None)
    s["show_stats"] = now < SHOW["until"]
    s["board"] = STATS.table() if s["show_stats"] else None
    s["swap"] = STATE.swap
    st = AIR.get("seen_t")                              # что видит главная камера (для пульта без камеры)
    s["camera_seen"] = dict(AIR["seen"], age=round(now - st, 1)) if st and AIR["cid"] else None
    s["match"] = {"active": MATCH["active"], "started": MATCH["started"], "set": s["us"]["sets"] + s["them"]["sets"] + 1,
                  "prompt": MATCH["prompt"], "warn": None,
                  "pit": ({"auto": pit["auto"], "since": pit["since"], "draft": pit["draft"], "before": pit.get("before"),
                           "touched": bool(pit.get("touched"))} if pit else None)}
    ms = MATCH.get("mirror_since")
    if MATCH["active"] and ms and now - ms > MIRROR_SEC:
        s["match"]["warn"] = "Похоже, стороны перепутаны: камера видит зеркальный счёт. Проверьте «Наши слева / справа»."
    elif MATCH.get("healed") and now - MATCH["healed"]["at"] < HEALED_SHOW_SEC:
        hs = MATCH["healed"]["score"]
        s["match"]["warn"] = (f"Питстоп снят сам: камера снова видит счёт {hs[0]}:{hs[1]}. "
                              "Пропущенные очки в статистике - полупрозрачные (порядок неизвестен).")
    elif RESTORE["at"] and now - RESTORE["at"] < RESTORED_SHOW_SEC:
        s["match"]["warn"] = ("Сервер перезапускался. Счёт восстановлен с телефона в "
                              + time.strftime("%H:%M", time.localtime(RESTORE["at"])) + " — проверьте его.")
    if pit:                                             # плашка: серый счёт, «Счёт уточняется», значки скрыты
        s["pit"] = True
        s["serve"] = None
        s["ball"] = s["streak"] = s["summary"] = None
    else:
        s["pit"] = False
    return s


def remember():
    with STATE.lock:
        UNDO.append({"points": list(STATE.points), "sets": list(STATE.sets), "points_ready": STATE.points_ready,
                     "sets_ready": STATE.sets_ready, "credited": STATE._credited, "winner": STATE._credit_winner,
                     "stats": STATS.dump()})
        del UNDO[:-UNDO_MAX]


def undo():
    if not UNDO:
        return False
    u = UNDO.pop()
    with STATE.lock:
        STATE.points, STATE.sets = u["points"], u["sets"]
        STATE.points_ready, STATE.sets_ready = u["points_ready"], u["sets_ready"]
        STATE._credited, STATE._credit_winner = u["credited"], u["winner"]
        STATE._pc, STATE._pn, STATE._sc, STATE._sn, STATE._fin_n = None, 0, None, 0, 0
        STATE._hold_p = time.time() + 15
        STATE._hold_s = time.time() + 60
        STATE.status = "последнее действие отменено"
        STATE.updated = time.time()
        STATE.save()
    STATS.restore(u["stats"])
    return True


def reset_set():
    """Счёт партии в 0:0, партии не трогаем. Если партия законно закончилась (25:21) - она уходит в статистику
    как завершённая; если нет - статистика текущей партии очищается."""
    with STATE.lock:
        STATE.points = [0, 0]
        STATE._pc, STATE._pn, STATE._fin_n = None, 0, 0
        STATE._credited, STATE._credit_winner = False, None
        STATE._hold_p = time.time() + 15
        STATE.status = "счёт партии сброшен"
        STATE.updated = time.time()
        STATE.save()
    target = STATE._target()
    if STATS._finished(tuple(STATS.cur), target):
        STATS.observe((0, 0), target, immediate=True)     # партия законно закончилась: сохраняем её как итог, а не стираем
    else:
        STATS.reset_set()
    SERVE.update(prev=(0, 0), who=None, hist=[])


def _us_them():
    sc = STATE.snapshot()
    return sc["us"]["points"], sc["them"]["points"]


def save_frame(data, mode, reading, took_ms, cid="anon", ignored=False):
    """Сохраняем кадр, когда показание изменилось, когда табло не читается (не чаще раза в 3 с) и просто раз в 20 с."""
    now = time.time()
    pts = (reading["left"]["points"], reading["right"]["points"]) if reading and reading.get("mode") else None
    unreadable = pts is None or None in pts
    changed = pts != FLOG["pts"]
    FLOG["pts"] = pts
    if not (changed or (unreadable and now - FLOG["saved"] >= 3) or now - FLOG["saved"] >= 20):
        return
    FLOG["saved"] = now
    flag = "LOST" if unreadable else ("chg" if changed else "smp")
    lr = "x-x" if unreadable and pts is None else f"{pts[0]}-{pts[1]}"
    name = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}_{mode}_{flag}_{lr}_{cid[:4]}.jpg"
    with open(os.path.join(FRAMES_DIR, name), "wb") as f:
        f.write(data)
    with open(os.path.join(FRAMES_DIR, "log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"file": name, "mode": mode, "flag": flag, "cid": cid, "ignored": ignored, "reading": reading,
                            "score": STATE.snapshot(), "took_ms": took_ms}, ensure_ascii=False) + "\n")
    files = sorted(x for x in os.listdir(FRAMES_DIR) if x.endswith(".jpg"))
    lost_crops = [x for x in files if "_crop_LOST_" in x]
    drop = lost_crops[:-MAX_LOST_CROPS]
    rest = [x for x in files if x not in drop]
    drop += rest[:-MAX_FRAMES] if len(rest) > MAX_FRAMES else []
    for old in drop:
        try:
            os.remove(os.path.join(FRAMES_DIR, old))
        except OSError:
            pass


EMPTY = {"mode": None, "left": {"points": None, "sets": None}, "right": {"points": None, "sets": None}}


def _readable(reading):
    return bool(reading.get("mode")) and reading["left"]["points"] is not None and reading["right"]["points"] is not None


def _sanitize(reading):
    """Проверка по правилам волейбола. Невозможные очки (77:9, 31:20) и партии (4:1) считаются непрочитанными:
    они не попадают ни на плашку, ни в статистику, ни в «камера видит». Работает и для самого первого показания."""
    if not reading or not reading.get("mode"):
        return reading
    r = json.loads(json.dumps(reading))
    L, R = r["left"], r["right"]
    if L["points"] is not None and R["points"] is not None and not valid_points(L["points"], R["points"], STATE._target()):
        L["points"] = R["points"] = None
    if L["sets"] is not None and R["sets"] is not None and not valid_sets(L["sets"], R["sets"], STATE.sets_to_win):
        L["sets"] = R["sets"] = None
    return r


def _seen(reading, swap):
    """Что видит камера, в терминах наши/соперник (по выбранной на этой странице стороне)."""
    if not reading or not reading.get("mode"):
        return None
    us, them = ("right", "left") if swap else ("left", "right")
    return {"us": reading[us]["points"], "them": reading[them]["points"],
            "us_sets": reading[us]["sets"], "them_sets": reading[them]["sets"]}


LASTS = {}                                     # по каждой странице: прошлый кадр (для "картинка та же") и показание


def process_frame(data, swap, mode="crop", cid="anon", air=False):
    """mode="full": весь кадр, табло ищется само; mode="crop": кадр уже обрезан вокруг найденного табло.
    air=True: страница считает себя в эфире. Счёт меняется, только если она главная и источник счёта - камера."""
    t0 = time.time()
    if cid == "anon":                       # страница без идентификатора - это старая версия, её кадры не берём
        return {"ok": False, "build": BUILD,
                "error": "Старая страница. Закройте эту вкладку и откройте ссылку заново"}
    if air and AIR["cid"] is None:          # эфир никто не держит (например, сервер перезапустился), а страница в эфире - она и главная
        AIR["cid"], AIR["last"] = cid, time.time()
        save_air()
    main = air and AIR["cid"] == cid
    evicted = air and not main              # эту страницу сняли с эфира: эфир взяла другая
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "build": BUILD, "error": "не картинка"}
    if float(img.mean()) < 3 and float(img.max()) < 25:      # камера отдала чёрный кадр: не сохраняем и в счёт не берём
        return {"ok": True, "build": BUILD, "black": True, "lost": True, "found": False, "bbox": None,
                "evicted": evicted, "seen": None, "score": payload(), "took_ms": 0}
    with open(os.path.join(DATA, "latest.jpg"), "wb") as f:
        f.write(data)
    with FRAME_LOCK:
        last = LASTS.setdefault(cid, {"gray": None, "reading": None})
        if len(LASTS) > 20:                                   # старые страницы забываем
            for k in list(LASTS)[:-20]:
                LASTS.pop(k, None)
        res = {"ok": True, "build": BUILD, "mode": mode, "evicted": evicted, "cached": False}
        swap = STATE.swap                                     # сторона наших - общая для всех страниц, хранится на сервере
        L, R = (STATE.points[0], STATE.points[1])
        recog.EXPECT = {"left": {L, L + 1}, "right": {R, R + 1}} if STATE.points_ready else {"left": set(), "right": set()}
        if mode == "full":
            found = find_board(img)
            last["gray"] = last["reading"] = None
            bbox, reading = found if found else (None, EMPTY)
            reading = _sanitize(reading)
            res.update(found=bool(found), bbox=bbox)
            readable = bool(found) and _readable(reading)
        else:
            # "Картинка та же" только если почти ни один пиксель не изменился заметно (смена одной цифры - доли процента).
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 64),
                               interpolation=cv2.INTER_AREA).astype(np.int16)
            unchanged = (last["gray"] is not None and last["reading"] is not None
                         and float((np.abs(small - last["gray"]) > 40).mean()) < 0.0015)
            if unchanged:
                reading = last["reading"]            # картинка та же: распознавание не гоняем, но кадр засчитываем
            else:
                reading = _sanitize(read_board(img))
                last["gray"], last["reading"] = small, reading
            readable = _readable(reading)
            res.update(cached=unchanged)
        if not SETTINGS["board_sets"] and reading.get("mode"):       # партий на табло нет: то, что похоже на них, не читаем
            reading = dict(reading, left=dict(reading["left"], sets=None), right=dict(reading["right"], sets=None))
        if main:
            AIR["last"] = time.time()
            if AIR["source"] == "camera" and not MATCH["pit"]:
                STATE.update(reading if readable or mode == "crop" else None)
            elif AIR["source"] == "camera" and MATCH["pit"] and MATCH["pit"].get("auto"):
                pit_heal(_seen(reading, swap) if readable else None)
            sn = _seen(reading, swap) if readable else None       # камера видит зеркальный счёт: стороны перепутаны?
            if sn:
                AIR["seen"], AIR["seen_t"] = sn, time.time()
            if sn and STATE.points_ready:
                u, t = STATE.snapshot()["us"]["points"], STATE.snapshot()["them"]["points"]
                if u != t and (sn["us"], sn["them"]) == (t, u):
                    MATCH["mirror_since"] = MATCH.get("mirror_since") or time.time()
                else:
                    MATCH["mirror_since"] = None
        took = int((time.time() - t0) * 1000)
        save_frame(data, mode, reading, took, cid, not main)
        res.update(lost=not readable, reading=reading, seen=_seen(reading, swap) if readable else None,
                   score=payload(), took_ms=took)
        return res


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _file(self, name, ctype):
        path = os.path.join(BASE, name)
        if not os.path.exists(path):
            return self._send(404, {"error": "нет файла " + name})
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return None
        return self.rfile.read(n)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._file("index.html", "text/html; charset=utf-8")
        elif path == "/version":
            self._send(200, {"build": BUILD})
        elif path == "/backup":
            self._send(200, backup())
        elif path == "/control":
            self._file("index.html", "text/html; charset=utf-8")         # пульт: та же страница, без камеры
        elif path == "/stats":
            self._file("stats.html", "text/html; charset=utf-8")
        elif path == "/stats.json":
            sc = payload()
            self._send(200, {"names": {"us": sc["us"]["name"], "them": sc["them"]["name"]}, "score": sc,
                             "table": STATS.table()})
        elif path == "/overlay":
            self._file("overlay.html", "text/html; charset=utf-8")
        elif path == "/score":
            self._send(200, payload())
        elif path == "/frames.zip":
            q = parse_qs(urlparse(self.path).query)
            if q.get("key", [""])[0] != FRAMES_KEY:
                return self._send(403, {"error": "нужен ключ: /frames.zip?key=..."})
            n = min(400, max(1, int(q.get("n", ["60"])[0])))
            flag = q.get("flag", ["all"])[0]                       # all | LOST | chg | smp
            files = sorted(x for x in os.listdir(FRAMES_DIR) if x.endswith(".jpg"))
            if flag != "all":
                files = [x for x in files if f"_{flag}_" in x]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for x in files[-n:]:
                    z.write(os.path.join(FRAMES_DIR, x), x)
                if os.path.exists(os.path.join(FRAMES_DIR, "log.jsonl")):
                    z.write(os.path.join(FRAMES_DIR, "log.jsonl"), "log.jsonl")
            self._send(200, buf.getvalue(), "application/zip", {"Content-Disposition": "attachment; filename=frames.zip"})
        elif path == "/latest.jpg":
            p = os.path.join(DATA, "latest.jpg")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            else:
                self._send(404, {"error": "кадров ещё нет"})
        else:
            self._send(404, {"error": "нет такого адреса"})

    def do_POST(self):
        u = urlparse(self.path)
        data = self._body()
        if data is None:
            return self._send(400, {"ok": False, "error": "пустое или слишком большое тело"})
        if u.path not in ("/frame", "/air", "/restore"):
            RESTORE["fresh"] = False                # счёт тронули руками: копию с телефона больше не принимаем
        try:
            if u.path == "/restore":
                with FRAME_LOCK:                    # пока восстанавливаем, кадры не обрабатываются
                    ok, msg = restore(json.loads(data))
                self._send(200, {"ok": ok, "message": msg, "score": payload()})
            elif u.path == "/frame":
                q = parse_qs(u.query)
                swap = q.get("swap", ["0"])[0] == "1"
                mode = "full" if q.get("mode", ["crop"])[0] == "full" else "crop"
                cid = (q.get("cid", ["anon"])[0] or "anon")[:12]
                air = q.get("air", ["0"])[0] == "1"
                self._send(200, process_frame(data, swap, mode, cid, air))
            elif u.path == "/air":
                j = json.loads(data)
                cid = str(j.get("cid", ""))[:12]
                if not cid:
                    return self._send(400, {"ok": False, "error": "нет cid"})
                if j.get("on"):
                    AIR["cid"], AIR["last"] = cid, time.time()          # эфир берёт эта страница, прежняя главная снимается
                elif AIR["cid"] == cid:
                    AIR["cid"] = None
                save_air()
                self._send(200, {"ok": True, "main": AIR["cid"] == cid, "score": payload()})
            elif u.path == "/source":
                j = json.loads(data)
                if j.get("source") not in ("camera", "manual"):
                    return self._send(400, {"ok": False, "error": "source: camera или manual"})
                AIR["source"] = j["source"]
                save_air()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/undo":
                done = undo()
                self._send(200, {"ok": True, "undone": done, "score": payload()})
            elif u.path == "/adjust":
                j = json.loads(data)
                if j.get("who") not in ("us", "them") or j.get("field") not in ("points", "sets") \
                        or j.get("delta") not in (1, -1):
                    return self._send(400, {"ok": False, "error": "неверные параметры"})
                if j["field"] == "points" and not j.get("force"):
                    sc = STATE.snapshot()
                    nu = sc["us"]["points"] + (j["delta"] if j["who"] == "us" else 0)
                    nt = sc["them"]["points"] + (j["delta"] if j["who"] == "them" else 0)
                    if not valid_points(nu, nt, STATE._target()):           # по правилам такого счёта не бывает: спросить
                        return self._send(200, {"ok": False, "confirm": f"Счёт {nu}:{nt} невозможен по правилам волейбола. Всё равно поставить?",
                                                "score": payload()})
                remember()
                STATE.adjust(j["who"], j["field"], j["delta"])
                if j["field"] == "points":
                    STATS.observe(_us_them(), STATE._target(), immediate=True)     # кнопки пишутся в статистику сразу
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/reset":
                remember()
                STATE.reset()
                MATCH.update(prev=[0, 0], pit=None, prompt=None, mid_swap_set=None)
                if MATCH["active"]:
                    with STATE.lock:
                        STATE.points_ready = True
                save_match()
                STATS.reset_match()
                SERVE.update(prev=(0, 0), who=None, hist=[])
                LASTS.clear()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/reset_set":
                remember()
                reset_set()
                LASTS.clear()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/close_set":
                remember()
                STATS.close_set()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/showstats":
                j = json.loads(data)
                SHOW["until"] = time.time() + SHOW_SEC if j.get("on") else 0.0
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/side":                        # «Наши слева / справа»: только указание, счёт не переставляется
                j = json.loads(data)
                with STATE.lock:
                    STATE.swap = bool(j.get("swap"))
                    STATE._pc, STATE._pn, STATE._sc, STATE._sn = None, 0, None, 0
                    STATE.save()
                MATCH["mirror_since"] = None
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/side_answer":                 # ответ на вопрос после конца партии
                j = json.loads(data)
                side_answer(j.get("us_left"))
                MATCH["mirror_since"] = None
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/match/start":
                j = json.loads(data)
                if j.get("us") or j.get("them"):
                    STATE.set_names(str(j.get("us", "")), str(j.get("them", "")))
                match_start()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/match/end":
                self._send(200, {"ok": True, "archived": match_end(), "score": payload()})
            elif u.path == "/set_end":                     # «Конец партии» вручную: партия победителю, счёт 0:0
                remember()
                us, th, us_s, th_s = _team_pts()
                if us != th:
                    w = 0 if us > th else 1
                    sets = [us_s + (w == 0), th_s + (w == 1)]
                    STATE.set_team_values(us, th, *sets)
                STATS.close_set()
                MATCH["prev"] = [max(us, 20), th]                 # чтобы смена сторон сработала на 0:0
                reset_set()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/pit/start":
                pit_start(auto=False)
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/pit/draft":
                j = json.loads(data)
                if MATCH["pit"]:
                    seq = [[int(t), int(q)] for t, q in j.get("seq", []) if int(t) in (0, 1)][:120]
                    sets = [max(0, min(STATE.sets_to_win, int(x))) for x in j.get("sets", MATCH["pit"]["draft"]["sets"])][:2]
                    MATCH["pit"]["draft"] = {"seq": seq, "sets": sets}
                    MATCH["pit"]["touched"] = True          # человек правит счёт: сам питстоп уже не снимется
                    save_match()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/pit/done":
                pit_done()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/pit/cancel":
                pit_cancel()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/settings":
                j = json.loads(data)
                if "serve" in j:
                    SETTINGS["serve"] = bool(j["serve"])
                if "board_sets" in j:
                    SETTINGS["board_sets"] = bool(j["board_sets"])
                    with STATE.lock:
                        STATE.sets_ready = False             # при «есть» табло подтвердит партии заново, при «нет» сервер считает сам
                        STATE._sc, STATE._sn = None, 0
                        STATE.save()
                save_air()
                self._send(200, {"ok": True, "score": payload()})
            elif u.path == "/names":
                j = json.loads(data)
                STATE.set_names(str(j.get("us", "")), str(j.get("them", "")))
                self._send(200, {"ok": True, "score": payload()})
            else:
                self._send(404, {"error": "нет такого адреса"})
        except Exception as e:                      # ошибка в одном запросе не должна валить сервер
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Сервер счёта запущен на порту {port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()

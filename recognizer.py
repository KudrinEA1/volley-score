"""Чтение счёта с табло. Бесплатно: OpenCV + Tesseract.

read_board(bgr) -> {"mode": "yellow" | "colored" | None,
                    "left":  {"points": int|None, "sets": int|None},
                    "right": {"points": int|None, "sets": int|None}}
Сторона left/right - физическая сторона табло в кадре, не "наши/соперник".

Поддерживаются три вида табло:
  mono    - без цвета: белые цифры на серых карточках, слева направо очки, партии, партии, очки;
  yellow  - жёлтые очки по краям, белые партии в центре на тёмном фоне;
  colored - цветные карточки (синяя и оранжевая), цифры белые;
            крупные цифры на карточке = очки, мелкие = партии.
"""
import time

import cv2
import numpy as np
import pytesseract

import digits as digit_net

WIDTH = 1000          # кадр приводится к этой ширине, пороги ниже рассчитаны на неё
MIN_CONF = 70         # ниже этой уверенности цифра считается нечитаемой
EARLY_CONF = 90       # при такой уверенности перебор вариантов останавливается
NET_CONF = 0.9        # нейросеть цифр: если уверена не меньше, ответ принимается сразу (без Tesseract)
READ_BUDGET = 1.0     # секунд на кадр: после этого медленный запасной Tesseract не запускается (кадр лучше пропустить)
SEARCH_BUDGET = 1.5   # то же для поиска табло по всему кадру
_BUDGET = {"until": 0.0, "outer": False}
# Подсказка ожидания: какие очки логично увидеть по сторонам табло (текущий счёт и +1). Ставит сервер перед чтением.
# Используется только при выборе между прочтениями: уверенно увиденное другое число не подменяется.
EXPECT = {"left": set(), "right": set()}
EXPECT_BONUS = 12      # прибавка к уверенности (шкала 0..100) для ожидаемого значения


def valid_points(a, b, target=25):
    """Счёт возможен по правилам волейбола: до target очков любой; выше - только с разницей не больше 2 (30:28, не 31:20)."""
    if a is None or b is None or a < 0 or b < 0:
        return False
    return max(a, b) <= target or abs(a - b) <= 2


def valid_sets(a, b, sets_to_win=3):
    return a is not None and b is not None and 0 <= a <= sets_to_win and 0 <= b <= sets_to_win and a + b <= 2 * sets_to_win - 1


# порядок вариантов: сначала без наклона (утоньшение жирных цифр), потом с наклоном (камера сбоку)
_VARIANTS = [(0, 0), (4, 0), (7, 0), (0, -0.12), (0, 0.12), (4, -0.12), (4, 0.12), (0, -0.25), (0, 0.25)]


def _ocr_tesseract(digits, limit=None):
    """Запасной путь: Tesseract. digits: белые цифры на чёрном. Возвращает (текст, уверенность 0..100).
    Перебираем варианты по порядку и останавливаемся на первом уверенном ответе (так быстрее)."""
    best = ("", -1)
    h = digits.shape[0]
    for erode, shear in _VARIANTS[:limit]:
        base = cv2.erode(digits, np.ones((3, 3), np.uint8), iterations=erode) if erode else digits
        pad = int(abs(shear) * h) + 40
        canvas = cv2.copyMakeBorder(base, 40, 40, pad, pad, cv2.BORDER_CONSTANT, value=0)
        hh, ww = canvas.shape
        m = np.float32([[1, shear, -shear * hh / 2], [0, 1, 0]])
        im = 255 - cv2.warpAffine(canvas, m, (ww, hh))
        im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        for psm in (7, 8):
            d = pytesseract.image_to_data(
                im, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789",
                output_type=pytesseract.Output.DICT)
            txt = "".join(t for t in d["text"] if t.strip())
            confs = [c for t, c in zip(d["text"], d["conf"]) if t.strip() and c >= 0]
            if txt and confs and np.mean(confs) > best[1]:
                best = (txt, float(np.mean(confs)))
            if best[1] >= EARLY_CONF:
                return best
    return best


def _tess_allowed():
    return time.time() < _BUDGET["until"]


def _ocr(digits):
    """digits: белые цифры на чёрном (одно число). Возвращает (текст, уверенность 0..100).
    Сначала нейросеть цифр (держит наклон и вид сбоку), если не уверена, то Tesseract; берётся более уверенный ответ."""
    net = digit_net.read_number(digits)
    if net and net[1] >= NET_CONF:
        return net[0], net[1] * 100
    if net and net[1] < 0.35:                                   # сеть уверенно "не цифра": мусор, Tesseract не тратим
        return net[0], net[1] * 100
    if not _tess_allowed():                                     # время на кадр вышло: медленный путь не запускаем
        return (net[0], net[1] * 100) if net else ("", -1)
    tess = _ocr_tesseract(digits, limit=3 if digit_net._net() else None)   # запасной путь; с сетью достаточно быстрого перебора
    if net and net[1] * 100 > tess[1]:
        return net[0], net[1] * 100
    return tess


def _ocr_options(digits, max_len):
    """Все правдоподобные прочтения числа очков: [(значение, уверенность 0..100)], уверенные первыми."""
    opts = [(t, c * 100) for t, c in digit_net.read_number_options(digits)]
    best = max((c for _, c in opts), default=-1)
    if not opts or 35 <= best < NET_CONF * 100:
        if _tess_allowed():
            opts.append(_ocr_tesseract(digits, limit=3 if digit_net._net() else None))
    out = []
    for t, c in sorted(opts, key=lambda r: -r[1]):
        v = _value((t, c), max_len)
        if v is not None and all(v != o[0] for o in out):
            out.append((v, c))
    return out


def _pick_points(opts_l, opts_r):
    """Пара очков, которая проходит правила (77:9 нельзя, 7:9 можно); из подходящих - самая уверенная."""
    best = None
    for a, ca in opts_l:
        for b, cb in opts_r:
            if valid_points(a, b):
                ca2 = ca + (EXPECT_BONUS if a in EXPECT["left"] else 0)
                cb2 = cb + (EXPECT_BONUS if b in EXPECT["right"] else 0)
                sc = min(ca2, cb2)
                if best is None or sc > best[0]:
                    best = (sc, a, b)
    return (best[1], best[2]) if best else (None, None)


def _value(res, max_len):
    """(текст, уверенность) -> int или None."""
    txt, conf = res
    if not txt or conf < MIN_CONF or len(txt) > max_len:
        return None
    return int(txt)


def _crop_digits(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _groups(mask, gap_frac=0.04, min_h_frac=0.08):
    """Выбрасывает мелкий мусор и склеивает цифры одной карточки. Возвращает маски групп слева направо."""
    h, w = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_HEIGHT] >= min_h_frac * h:
            keep[lab == i] = 255
    merged = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(gap_frac * w)), 3)))
    n, lab, _, _ = cv2.connectedComponentsWithStats(merged)
    out = []
    for i in range(1, n):
        g = cv2.bitwise_and(keep, (lab == i).astype(np.uint8) * 255)
        if g.any():
            x, _, _, _ = cv2.boundingRect(g)
            crop = _crop_digits(g)
            if crop is not None:
                out.append((x, crop))
    return [c for _, c in sorted(out, key=lambda t: t[0])]


def _strong_yellow(groups, bgr, mask):
    """Оставляет группы, жёлтые "по-настоящему" (краска цифр), а не желтоватые стол, пол, кожу.
    Мера: медиана (R+G)/2 - B по пикселям маски; цифры ~150-250, мебель и паркет ~50-100."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    yl = (r + g) // 2 - b
    vals = []
    for x0, y0, x1, y1 in groups:
        m = mask[y0:y1, x0:x1] > 0
        vals.append(float(np.median(yl[y0:y1, x0:x1][m])) if m.any() else 0.0)
    if not vals:
        return []
    top = max(vals)
    return [gr for gr, v in zip(groups, vals) if v >= 90 and v >= 0.6 * top]


def _yellow_pair(yellow, bgr=None):
    """Две самые крупные жёлтые группы цифр в одну строку (очки слева и справа). Мусор (жёлтые предметы) отбрасывается."""
    H = yellow.shape[0]
    groups = _yellow_digit_groups(yellow, min_h=0.12, gap_max=0.6, max_aspect=2.6)
    if bgr is not None:
        groups = _strong_yellow(groups, bgr, yellow)
    best = None
    for i in range(len(groups)):
        for j in range(len(groups)):
            a, b = groups[i], groups[j]
            if b[0] <= a[2]:
                continue
            ha, hb = a[3] - a[1], b[3] - b[1]
            hm = max(ha, hb)
            if not 0.55 <= ha / hb <= 1.8 or abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) > 0.8 * hm or b[0] - a[2] > 6 * hm:
                continue
            score = min(ha, hb) * (min(ha, hb) / hm) ** 2
            if best is None or score > best[0]:
                best = (score, a, b)
    return None if best is None else best[1:]


def _sets_comps(binary, sat, hp):
    m = binary.astype(np.uint8) * 255
    m[sat > 90] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    hh, ww = m.shape
    comps = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in st[i]]
        if x <= 0 or y <= 0 or x + w >= ww or y + h >= hh:          # край карточки / фон
            continue
        if not 0.2 * hp <= h <= 0.95 * hp or w > 1.6 * h or area < 0.2 * w * h or area > 0.9 * w * h:
            continue
        comps.append((x, y, w, h, i))
    if comps:
        hmax = max(c[3] for c in comps)
        comps = [c for c in comps if c[3] >= 0.75 * hmax]           # цифры партий одной высоты; мелочь - мусор
    return comps, lab


def _read_sets_between(bgr, a, b):
    """Партии: светлые цифры на карточках между двумя группами очков. Порог Оцу только по этой полосе."""
    hp = min(a[3] - a[1], b[3] - b[1])
    y0, y1 = max(0, min(a[1], b[1]) - int(0.1 * hp)), min(bgr.shape[0], max(a[3], b[3]) + int(0.1 * hp))
    x0, x1 = a[2] + int(0.05 * hp), b[0] - int(0.05 * hp)
    if x1 - x0 < 0.3 * hp:
        return None, None
    roi = bgr[y0:y1, x0:x1]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
    t1, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = gray[gray > t1]
    t2 = cv2.threshold(bright.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if bright.size > 50 else t1
    # t1 отделяет светлое от тёмного; если карточки сами светло-серые, цифры отделяет второй порог t2
    for t in (t1, t2):
        comps, lab = _sets_comps(gray > t, sat, hp)
        if len(comps) == 2:
            break
    else:
        return None, None
    out = []
    for x, y, w, h, i in sorted(comps):
        crop = _crop_digits((lab == i).astype(np.uint8) * 255)
        out.append(None if crop is None else _value(_ocr(crop), 1))
    return out[0], out[1]


def _read_yellow(hsv, yellow, bgr=None):
    pair = _yellow_pair(yellow, bgr)
    if pair is None:
        return (None, None), (None, None)
    a, b = pair
    bb, gg, rr = [c.astype(np.int16) for c in cv2.split(bgr)] if bgr is not None else (None, None, None)
    pts = []
    for x0, y0, x1, y1 in (a, b):
        part = yellow[y0:y1, x0:x1].copy()
        if bgr is not None:                     # внутри группы выкидываем куски, которые желтее фона, но не краска цифр
            yl = (rr[y0:y1, x0:x1] + gg[y0:y1, x0:x1]) // 2 - bb[y0:y1, x0:x1]
            n, lab, st, _ = cv2.connectedComponentsWithStats(part)
            comps = [(i, float(np.median(yl[lab == i])), int(st[i, 3])) for i in range(1, n) if st[i, 4] >= 12]
            if comps:
                top, hmax = max(c[1] for c in comps), max(c[2] for c in comps)
                for i, v, h in comps:
                    if v < 0.6 * top or h < 0.45 * hmax:
                        part[lab == i] = 0
        crop = _crop_digits(part)
        pts.append([] if crop is None else _ocr_options(crop, 2))
    points = _pick_points(pts[0], pts[1])
    sets = _read_sets_between(bgr, a, b) if bgr is not None else (None, None)
    if None in sets:
        sets = _read_sets_dark(hsv)
    return points, sets


def _read_sets_dark(hsv):
    # запасной путь для партий: белые цифры, окружённые тёмным (то есть лежащие на тёмной карточке)
    white = cv2.inRange(hsv, (0, 0, 170), (179, 70, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    hh, ww = white.shape
    on_card = np.zeros_like(white)
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        if h < 0.08 * hh:
            continue
        m = int(0.25 * h)
        x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(ww, x + w + m), min(hh, y + h + m)
        around = hsv[y0:y1, x0:x1, 2][(lab[y0:y1, x0:x1] != i) & (white[y0:y1, x0:x1] == 0)]
        if around.size and np.median(around) < 80:
            on_card[lab == i] = 255
    sts = [_value(_ocr(g), 1) for g in _groups(on_card, gap_frac=0.03)]
    sts = [v for v in sts if v is not None]            # мусор с нулевой уверенностью отбрасываем
    return (sts[0], sts[1]) if len(sts) == 2 else (None, None)


def _region(color_mask):
    m = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 0.004 * m.size]
    if not cnts:
        return None, None
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    ys, xs = np.where(filled > 0)
    return cv2.erode(filled, np.ones((7, 7), np.uint8)), float(xs.mean())


def _read_team_color(hsv, color_mask):
    """Возвращает (очки, партии, x-центр) для команды одного цвета или None."""
    region, cx = _region(color_mask)
    if region is None:
        return None
    white = cv2.inRange(hsv, (0, 0, 170), (179, 80, 255))
    white = cv2.bitwise_and(white, region)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(white)
    comps = [(st[i, 3], i) for i in range(1, n) if st[i, 4] > 150]       # (высота, номер)
    if not comps:
        return None
    hmax, hmin = max(c[0] for c in comps), min(c[0] for c in comps)
    split = (hmax + hmin) / 2 if hmax / max(hmin, 1) > 1.25 else 0       # крупные = очки, мелкие = партии

    def read(sel, max_len):
        if not sel:
            return None
        m = np.zeros(white.shape, np.uint8)
        for _, i in sel:
            m[lab == i] = 255
        crop = _crop_digits(m)
        return None if crop is None else _value(_ocr(crop), max_len)

    points = read([c for c in comps if c[0] > split], 2)
    sets = read([c for c in comps if c[0] <= split], 1) if split else None
    return points, sets, cx



def _yellow_mask(bgr, hsv):
    """"Жёлтость" = (красный + зелёный)/2 - синий: почти не зависит от яркости, поэтому цифры в тени тоже находятся."""
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    hue = hsv[:, :, 0]                                              # оттенок жёлтого, оранжевые карточки сюда не попадают
    y = (((r + g) // 2 - b > 45) & (hsv[:, :, 2] > 110) & (hue >= 15) & (hue <= 50)).astype(np.uint8) * 255
    y = cv2.morphologyEx(y, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(y, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


# ====================== табло без цвета: белые цифры на тёмно-серых карточках ======================
# слева направо: очки левых, партии левых, партии правых, очки правых (у табло без партий - просто две карточки)
def _mono_mask(bgr):
    """Светлые малонасыщенные пятна, заметно ярче окружения (цифры на карточке)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    k = max(31, (bgr.shape[1] // 12) | 1)
    bg = cv2.blur(gray.astype(np.uint8), (k, k)).astype(np.int16)
    # ярче окружения ИЛИ просто очень светлое (большие цифры сами поднимают средний фон под собой)
    m = (((gray - bg > 35) | (gray > 170)) & (gray > 120) & (sat < 90)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _on_dark_card(gray, box):
    """Светлая группа лежит на тёмной карточке: вокруг заметно темнее, чем сами цифры (отсекает кожу, надписи, блики)."""
    x0, y0, x1, y1 = box
    H, W = gray.shape
    p = int(0.25 * (y1 - y0))
    a, b, c, d = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
    win = gray[b:d, a:c]
    inner = np.zeros(win.shape, bool)
    inner[y0 - b:y1 - b, x0 - a:x1 - a] = True
    ring = win[~inner]
    digits = gray[y0:y1, x0:x1]
    if ring.size < 20:
        return False
    r, v = float(np.percentile(ring, 30)), float(np.percentile(digits, 85))
    return r < 120 and v - r > 70


def _mono_runs(groups, size_wh=None, gray=None):
    """Группы цифр -> варианты "табло в ряд": (оценка, [группы слева направо]). Четвёрка (с партиями) лучше пары."""
    groups = sorted(groups)
    out = []
    for i, a in enumerate(groups):
        ha = a[3] - a[1]
        row = [a]
        for g in groups[i + 1:]:
            hg = g[3] - g[1]
            if abs((g[1] + g[3]) / 2 - (a[1] + a[3]) / 2) <= 0.45 * ha and 0.45 <= hg / ha <= 1.8 and g[0] - row[-1][2] >= 0.1 * ha:
                row.append(g)
        for size in (4,):                      # пара без карточек партий слишком похожа на случайные светлые цифры (надписи, майки)
            if len(row) < size:
                continue
            run = row[:size]
            if size_wh and (run[0][0] <= 2 or run[-1][2] >= size_wh[0] - 2 or min(g[1] for g in run) <= 2 or max(g[3] for g in run) >= size_wh[1] - 2):
                continue                                           # табло обрезано краем кадра: цифры на краю читаются неверно
            hs = [g[3] - g[1] for g in run]
            hm = max(hs)
            gaps = [run[k + 1][0] - run[k][2] for k in range(size - 1)]
            if max(gaps) > 3 * hm or min(gaps) < 0.15 * hm:
                continue
            outer = (hs[0], hs[-1])
            if not 0.65 <= outer[0] / outer[1] <= 1.54:
                continue
            if size == 4 and not all(0.45 * hm <= h <= 1.15 * hm for h in hs[1:3]):
                continue
            dark = sum(_on_dark_card(gray, g) for g in run) if gray is not None else 0       # сколько карточек лежат на тёмном
            out.append((min(outer) * (2 if size == 4 else 1) * (1 + dark / 2.0), run))
    out.sort(key=lambda t: -t[0])
    return out


def _mono_candidates(img):
    """Рамки (x0, y0, x1, y1) вокруг рядов светлых карточек, лучшие первыми."""
    h_img, w_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    groups = _yellow_digit_groups(_mono_mask(img), min_h=0.02, gap_max=0.2)
    boxes = []
    for _, run in _mono_runs(groups, (w_img, h_img), gray):
        x0, y0 = min(g[0] for g in run), min(g[1] for g in run)
        x1, y1 = max(g[2] for g in run), max(g[3] for g in run)
        hm = y1 - y0
        mx, my = int(0.5 * hm), int(0.5 * hm)
        boxes.append((max(0, x0 - mx), max(0, y0 - my), min(w_img, x1 + mx), min(h_img, y1 + my)))
    return boxes


def _mono_glyph(gray, box):
    """Точная маска цифр одной карточки: порог Оцу внутри рамки (рамка чуть шире, чтобы не обрезать края цифр).
    Возвращает белые цифры на чёрном или None."""
    x0, y0, x1, y1 = box
    p = int(0.12 * (y1 - y0))
    H, W = gray.shape
    a, b, c, d = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
    g = gray[b:d, a:c]
    if g.size == 0:
        return None
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(0.06 * (y1 - y0)) | 1)                                   # тонкие перемычки к бликам и кольцам рвём
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    hh, ww = m.shape
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in st[i]]
        touches = x <= 0 or y <= 0 or x + w >= ww or y + h >= hh              # край карточки, а не цифра
        if not touches and h >= 0.3 * (y1 - y0) and area >= 30:
            keep[lab == i] = 255
    return _crop_digits(keep)


def _read_mono(bgr):
    mask = _mono_mask(bgr)
    groups = _yellow_digit_groups(mask, min_h=0.04, gap_max=0.2)
    runs = _mono_runs(groups, (bgr.shape[1], bgr.shape[0]), cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if not runs:
        return None
    run = runs[0][1]
    vals = []
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    for k, (x0, y0, x1, y1) in enumerate(run):
        crop = _mono_glyph(gray, (x0, y0, x1, y1))
        if crop is None:
            crop = _crop_digits(mask[y0:y1, x0:x1])
        outer = k in (0, len(run) - 1)
        if outer:
            vals.append([] if crop is None else _ocr_options(crop, 2))
        else:
            vals.append(None if crop is None else _value(_ocr(crop), 1))
    lp, rp = _pick_points(vals[0], vals[-1])
    if len(run) == 4:
        return {"mode": "mono", "left": {"points": lp, "sets": vals[1]}, "right": {"points": rp, "sets": vals[2]}}
    return {"mode": "mono", "left": {"points": lp, "sets": None}, "right": {"points": rp, "sets": None}}


# ====================== табло из белых карточек с тёмными или цветными цифрами (синее перекидное) ======================
# слева большая карточка (очки левых, например зелёные), справа большая (очки правых, чёрные), между ними маленькие
# карточки партий (красные). Партии отсюда не читаются.
def _white_mask(bgr, hsv=None):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if hsv is None else hsv
    m = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 150)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _white_boards(bgr, hsv=None):
    """Белые пятна табло (карточки стоят вплотную и сливаются): [(рамка x0, y0, x1, y1, залитый контур)], крупные первыми."""
    H, W = bgr.shape[:2]
    m = _white_mask(bgr, hsv)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.01 * W * H or w < 1.2 * h or cv2.contourArea(c) < 0.4 * w * h:
            continue
        fill = np.zeros((H, W), np.uint8)
        cv2.drawContours(fill, [c], -1, 255, -1)
        out.append(((x, y, x + w, y + h), fill))
    return out


def _board_groups(bgr, hsv, box, fill):
    """Цифры внутри белого табло -> группы (числа) слева направо: [(x0, y0, x1, y1, маска)] только самые высокие (очки)."""
    x0, y0, x1, y1 = box
    g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    s = hsv[y0:y1, x0:x1, 1]
    inside = fill[y0:y1, x0:x1] > 0
    paper = float(np.percentile(g[inside], 70)) if inside.any() else 200.0
    ink = ((((g < paper - 70) | (s > 90)) & inside).astype(np.uint8)) * 255       # тёмное или цветное на белом
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    hh = y1 - y0
    comps = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in st[i]]
        if h < 0.2 * hh or w < 0.12 * h or w > 1.2 * h or area < 60:
            continue                                                # мелочь, тени между карточками, полосы
        if area / h < 0.07 * h:
            continue                                                # тонкая линия (край карточки, тень), даже наклонная
        if y <= 1 and y + h >= hh - 1:
            continue
        comps.append([x, y, x + w, y + h, i])
    if not comps:
        return []
    hmax = max(c[3] - c[1] for c in comps)
    tall = sorted(c for c in comps if c[3] - c[1] >= 0.75 * hmax)    # очки - самые крупные цифры; партии мельче
    groups = []
    for c in tall:
        if groups and c[0] - groups[-1][2] <= 0.35 * (c[3] - c[1]):
            gr = groups[-1]
            gr[:4] = [min(gr[0], c[0]), min(gr[1], c[1]), max(gr[2], c[2]), max(gr[3], c[3])]
            gr[4].append(c[4])
        else:
            groups.append([c[0], c[1], c[2], c[3], [c[4]]])
    out = []
    for gx0, gy0, gx1, gy1, ids in groups:
        m = np.isin(lab[gy0:gy1, gx0:gx1], ids).astype(np.uint8) * 255
        out.append((gx0, gy0, gx1, gy1, m))
    return out


def _read_cards(bgr, hsv):
    for box, fill in _white_boards(bgr, hsv):
        groups = _board_groups(bgr, hsv, box, fill)
        if len(groups) < 2:
            continue
        vals = []
        for gr in (groups[0], groups[-1]):
            crop = _crop_digits(gr[4])
            vals.append([] if crop is None else _ocr_options(crop, 2))
        lp, rp = _pick_points(vals[0], vals[1])
        return {"mode": "cards", "left": {"points": lp, "sets": None}, "right": {"points": rp, "sets": None}}
    return None


def _cards_candidates(img, hsv):
    h_img, w_img = img.shape[:2]
    out = []
    for (x0, y0, x1, y1), _ in _white_boards(img, hsv)[:1]:
        mx, my = int(0.06 * (x1 - x0)), int(0.15 * (y1 - y0))
        out.append((max(0, x0 - mx), max(0, y0 - my), min(w_img, x1 + mx), min(h_img, y1 + my)))
    return out


def read_board(bgr):
    if not _BUDGET["outer"]:
        _BUDGET["until"] = time.time() + READ_BUDGET
    h, w = bgr.shape[:2]
    if w != WIDTH:
        bgr = cv2.resize(bgr, (WIDTH, max(1, int(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    empty = {"mode": None, "left": {"points": None, "sets": None}, "right": {"points": None, "sets": None}}

    def ok(r):
        return bool(r) and r["left"]["points"] is not None and r["right"]["points"] is not None

    first = None                                                    # первый вид, который хотя бы опознан (для отчёта, если ничего не прочиталось)
    yellow = _yellow_mask(bgr, hsv)
    if yellow.mean() / 255 > 0.008:
        points, sets = _read_yellow(hsv, yellow, bgr)
        r = {"mode": "yellow",
             "left": {"points": points[0], "sets": sets[0]},
             "right": {"points": points[1], "sets": sets[1]}}
        if ok(r):
            return r
        first = r

    blue = cv2.inRange(hsv, (100, 120, 60), (130, 255, 255))
    orange = cv2.inRange(hsv, (0, 120, 90), (20, 255, 255)) | cv2.inRange(hsv, (170, 120, 90), (179, 255, 255))
    teams = [_read_team_color(hsv, m) for m in (blue, orange)]
    if all(teams):
        left, right = sorted(teams, key=lambda t: t[2])                  # сторона определяется по положению цвета
        r = {"mode": "colored",
             "left": {"points": left[0], "sets": left[1]},
             "right": {"points": right[0], "sets": right[1]}}
        if ok(r):
            return r
        first = first or r

    r = _read_mono(bgr)
    if ok(r):
        return r
    first = first or r
    r = _read_cards(bgr, hsv)
    if ok(r):
        return r
    return first or r or empty


# ====================== автоматический поиск табло в кадре ======================
SEARCH_WIDTH = 1920      # кадр для поиска уменьшается до этой ширины (мелкое табло издалека требует больше пикселей)
MAX_TRIES = 2            # сколько самых вероятных кандидатов проверяем распознаванием (каждый - до нескольких секунд)


def _yellow_digit_groups(mask, min_h=0.03, gap_max=0.6, max_aspect=1.3):
    """Светлые "цифры" (по форме) -> группы рядом стоящих цифр [x0, y0, x1, y1]. Сплошные пятна отбрасываются.
    min_h - наименьшая высота цифры в долях кадра; gap_max - наибольший зазор между цифрами одной группы в долях высоты."""
    h_img, w_img = mask.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if h < min_h * h_img or h > 0.6 * h_img or w > max_aspect * h:      # слишком мелкое, огромное или не похоже на цифру
            continue
        if a < 0.25 * w * h or a > 0.85 * w * h:                    # у цифры есть "пустоты"; сплошной прямоугольник - не цифра
            continue
        comps.append([x, y, x + w, y + h])
    comps.sort()
    groups = []
    for c in comps:
        ch = c[3] - c[1]
        for g in groups:
            gh = g[3] - g[1]
            gap = c[0] - g[2]
            same_row = abs((c[1] + c[3]) / 2 - (g[1] + g[3]) / 2) <= 0.3 * gh
            if -0.45 * gh <= gap <= gap_max * gh and 0.7 <= ch / gh <= 1.4 and same_row:      # курсив и наклон: рамки цифр заходят друг на друга
                g[:] = [min(g[0], c[0]), min(g[1], c[1]), max(g[2], c[2]), max(g[3], c[3])]
                break
        else:
            groups.append(c[:])
    return groups


def _yellow_candidates(img, hsv):
    """Пары групп цифр: похожая высота, одна строка, разумное расстояние. Возвращает рамки (x0, y0, x1, y1), лучшие первыми."""
    h_img, w_img = img.shape[:2]
    ymask = _yellow_mask(img, hsv)
    groups = _strong_yellow(_yellow_digit_groups(ymask, min_h=0.02, max_aspect=2.6), img, ymask)
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            ha, hb = a[3] - a[1], b[3] - b[1]
            hm = max(ha, hb)
            gap = b[0] - a[2]
            same_row = abs((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) <= 0.8 * hm          # под углом табло "заваливается"
            if 0.55 <= ha / hb <= 1.8 and same_row and 0.3 * hm <= gap <= 6 * hm:
                x0, y0, x1, y1 = min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])
                mx, my = int(0.6 * hm), int(0.5 * hm)
                pairs.append((min(ha, hb) * (min(ha, hb) / hm) ** 2, (max(0, x0 - mx), max(0, y0 - my), min(w_img, x1 + mx), min(h_img, y1 + my))))
    pairs.sort(key=lambda t: -t[0])
    return [p[1] for p in pairs]


def _color_candidates(img, hsv):
    """Синяя и оранжевая карточки рядом -> рамка вокруг обеих."""
    h_img, w_img = img.shape[:2]
    blue = cv2.inRange(hsv, (100, 120, 60), (130, 255, 255))
    orange = cv2.inRange(hsv, (0, 120, 90), (20, 255, 255)) | cv2.inRange(hsv, (170, 120, 90), (179, 255, 255))

    def boxes(mask):
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted([c for c in cnts if cv2.contourArea(c) > 0.004 * h_img * w_img], key=cv2.contourArea, reverse=True)[:3]
        return [cv2.boundingRect(c) for c in cnts]

    out = []
    for bx, by, bw, bh in boxes(blue):
        for ox, oy, ow, oh in boxes(orange):
            hm, wm = max(bh, oh), max(bw, ow)
            same_row = abs((by + bh / 2) - (oy + oh / 2)) <= 0.8 * hm
            gap = max(bx, ox) - min(bx + bw, ox + ow)
            if same_row and gap <= 1.2 * wm:
                x0, y0 = min(bx, ox), min(by, oy)
                x1, y1 = max(bx + bw, ox + ow), max(by + bh, oy + oh)
                mx, my = int(0.05 * (x1 - x0)), int(0.05 * (y1 - y0))
                out.append(((x1 - x0) * (y1 - y0), (max(0, x0 - mx), max(0, y0 - my), min(w_img, x1 + mx), min(h_img, y1 + my))))
    out.sort(key=lambda t: -t[0])
    return [o[1] for o in out]


def find_board(bgr):
    """Ищет табло во всём кадре. Возвращает (рамка в долях кадра [x, y, w, h], показание) или None."""
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, SEARCH_WIDTH / w0)
    img = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    candidates = (_yellow_candidates(img, hsv)[:MAX_TRIES] + _color_candidates(img, hsv)[:1] + _mono_candidates(img)[:MAX_TRIES]
                  + _cards_candidates(img, hsv))
    _BUDGET["until"], _BUDGET["outer"] = time.time() + SEARCH_BUDGET, True       # общее время на весь поиск
    try:
        for x0, y0, x1, y1 in candidates:
            r = read_board(img[y0:y1, x0:x1])
            if r["mode"] and r["left"]["points"] is not None and r["right"]["points"] is not None:
                return [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h], r
        return None
    finally:
        _BUDGET["outer"] = False

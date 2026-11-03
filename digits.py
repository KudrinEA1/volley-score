"""Чтение цифр табло маленькой нейросетью. Работает под углом, потому что обучена на цифрах разных шрифтов,
которые искусственно поворачивали, наклоняли и сжимали (см. train_digits.py).

read_number(mask) -> (текст, уверенность 0..1) или None
  mask: белые цифры на чёрном, одна группа (одно число, 1-3 цифры)
Нужны только numpy и OpenCV; сеть лежит в digit_model.npz рядом с этим файлом.
"""
import os

import cv2
import numpy as np

_STATE = {"path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.npz"), "net": None}
_HOG = cv2.HOGDescriptor((24, 32), (16, 16), (8, 8), (8, 8), 9)


def feature(mask):
    """Одна цифра (белая на чёрном, обрезана по рамке) -> вектор признаков. Ширина и высота приводятся к стандартным,
    поэтому сжатие по горизонтали (вид сбоку) почти не влияет."""
    m = cv2.resize(mask, (24, 32), interpolation=cv2.INTER_AREA)
    hog = _HOG.compute(m).flatten()
    pix = cv2.resize(m, (12, 16), interpolation=cv2.INTER_AREA).flatten() / 255.0
    return np.concatenate([hog, pix]).astype(np.float32)


def tight(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _load_arrays():
    if os.path.exists(_STATE["path"]):
        return np.load(_STATE["path"])
    try:                                                    # текстовая копия модели (digit_model_b64.py)
        import base64
        import io
        import digit_model_b64
        return np.load(io.BytesIO(base64.b64decode(digit_model_b64.DATA)))
    except Exception:
        return None


def _net():
    if _STATE["net"] is None:
        d = _load_arrays()
        if d is None:
            _STATE["net"] = False
        else:
            n = int(d["n"])
            _STATE["net"] = {"layers": [(d[f"W{i}"].astype(np.float32), d[f"b{i}"].astype(np.float32)) for i in range(n)],
                             "mean": d["mean"].astype(np.float32), "std": d["std"].astype(np.float32)}
    return _STATE["net"]


def set_model(path):
    """Для проверок: подменить файл модели."""
    _STATE["path"], _STATE["net"] = path, None


def predict(features):
    """features: (N, D) -> вероятности (N, 10)"""
    net = _net()
    x = (np.asarray(features, np.float32) - net["mean"]) / net["std"]
    for W, b in net["layers"][:-1]:
        x = np.maximum(0, x @ W + b)
    W, b = net["layers"][-1]
    z = x @ W + b
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def split_digits(mask, cut=True):
    """Группа цифр -> список масок отдельных цифр слева направо. Обломки одной цифры склеиваются, слипшиеся цифры делятся."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    pieces = [{"x0": int(st[i, 0]), "x1": int(st[i, 0] + st[i, 2]), "y0": int(st[i, 1]), "y1": int(st[i, 1] + st[i, 3]),
               "ids": [i]} for i in range(1, n) if st[i, 4] >= 12]
    pieces.sort(key=lambda p: p["x0"])
    digits = []
    for p in pieces:
        for d in digits:
            overlap = min(p["x1"], d["x1"]) - max(p["x0"], d["x0"])
            hp, hd = p["y1"] - p["y0"], d["y1"] - d["y0"]
            yover = min(p["y1"], d["y1"]) - max(p["y0"], d["y0"])
            # обломок над/под цифрой: заметно ниже соседа или лежит выше/ниже его. Две полноразмерные наклонные цифры
            # ("1" заходит на "2" под углом) не склеиваем
            fragment = min(hp, hd) < 0.6 * max(hp, hd) or yover < 0.5 * min(hp, hd)
            if overlap >= 0.5 * min(p["x1"] - p["x0"], d["x1"] - d["x0"]) and fragment:
                d["x0"], d["x1"] = min(d["x0"], p["x0"]), max(d["x1"], p["x1"])
                d["y0"], d["y1"] = min(d["y0"], p["y0"]), max(d["y1"], p["y1"])
                d["ids"] += p["ids"]
                break
        else:
            digits.append(p)
    if not digits:
        return []
    hmax = max(d["y1"] - d["y0"] for d in digits)
    digits = [d for d in digits if d["y1"] - d["y0"] >= 0.45 * hmax]             # мелкий мусор
    out = []
    for d in sorted(digits, key=lambda d: d["x0"]):
        m = tight(np.isin(lab, d["ids"]).astype(np.uint8) * 255)
        h, w = m.shape
        if cut and w / h > 1.1 and w >= 8:                                              # две слипшиеся цифры: режем по самому узкому месту
            cols = (m > 0).sum(axis=0).astype(float)
            lo, hi = int(0.3 * w), int(0.7 * w)
            cut = lo + int(np.argmin(cols[lo:hi])) if hi > lo else w // 2
            for part in (m[:, :cut], m[:, cut:]):
                t = tight(part)
                if t is not None:
                    out.append(t)
        else:
            out.append(m)
    return out


def read_number_options(mask, max_digits=3):
    """Все прочтения группы: широкую цифру можно понять и как две слипшиеся ("77"), и как одну растянутую ("7").
    Список (текст, уверенность 0..1), без повторов."""
    if not _net():
        return []
    out = []
    for cut in (True, False):
        glyphs = split_digits(mask, cut)
        if not glyphs or len(glyphs) > max_digits:
            continue
        probs = predict(np.stack([feature(g) for g in glyphs]))
        res = ("".join(str(int(p.argmax())) for p in probs), float(probs.max(axis=1).min()))
        if all(r[0] != res[0] for r in out):
            out.append(res)
    return out


def read_number(mask, max_digits=3):
    """Группа цифр -> (текст, уверенность 0..1) или None: самое уверенное прочтение
    (разбиение на две цифры берётся, если оно не хуже цельного больше чем на 0.05)."""
    opts = read_number_options(mask, max_digits)
    if not opts:
        return None
    best = opts[0]
    for r in opts[1:]:
        if r[1] > best[1] + 0.05:
            best = r
    return best

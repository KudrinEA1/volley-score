"""Обучение распознавателя цифр (запускается не на сервере, а у меня здесь; на сервер идёт только digit_model.npz).

Данные делаются искусственно: цифры разных шрифтов и цифры, вырезанные с реальных табло, поворачиваются вбок
(вид сбоку), сверху/снизу, наклоняются, сжимаются, размываются и "огрубляются" до масок, какими их видит программа.
Нужны: numpy, OpenCV, Pillow, scikit-learn.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import digits as D

FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",       # как Arial
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf",   # как Helvetica
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheroscn-bold.otf", # Helvetica узкий
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/opentype/tlwg/Loma-Bold.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/texmf/fonts/opentype/public/lm/lmsans10-bold.otf",
    "/usr/share/texmf/fonts/opentype/public/lm/lmsansdemicond10-regular.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
]


def render_glyph(ch, font_path, size=140):
    font = ImageFont.truetype(font_path, size)
    img = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 4), ch, font=font, fill=255)
    return D.tight((np.array(img) > 128).astype(np.uint8) * 255)


def warp_3d(img, yaw, pitch, roll, shear, dist):
    """Плоская цифра, повёрнутая в пространстве (вид сбоку/сверху), с наклоном шрифта."""
    h, w = img.shape
    X = np.array([-w / 2, w / 2, w / 2, -w / 2], float)
    Y = np.array([-h / 2, -h / 2, h / 2, h / 2], float)
    X = X - shear * Y
    ay, ax, az = np.deg2rad([yaw, pitch, roll])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    P = np.stack([X, Y, np.zeros(4)], 1) @ (Rz @ Rx @ Ry).T
    Dd = dist * max(w, h)
    x = Dd * P[:, 0] / (Dd + P[:, 2])
    y = Dd * P[:, 1] / (Dd + P[:, 2])
    dst = np.stack([x - x.min(), y - y.min()], 1).astype(np.float32)
    size = (int(np.ceil(dst[:, 0].max())) + 1, int(np.ceil(dst[:, 1].max())) + 1)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), size)


def augment(glyph, rng):
    """Один случайный вариант цифры, как её увидит программа. Возвращает признаки или None."""
    g = glyph
    h, w = g.shape
    g = cv2.resize(g, (max(4, int(w * rng.uniform(0.5, 1.15))), h))                  # узкий/широкий шрифт
    r = int(rng.integers(-3, 6))                                                       # толщина штриха
    if r > 0:
        g = cv2.dilate(g, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    elif r < 0:
        g = cv2.erode(g, np.ones((2 * -r + 1, 2 * -r + 1), np.uint8))
    if g.sum() == 0:
        return None
    g = cv2.copyMakeBorder(g, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    near_front = rng.random() < 0.3
    yaw = rng.uniform(-8, 8) if near_front else float(np.clip(rng.normal(0, 38), -72, 72))
    pitch = float(np.clip(rng.normal(0, 15), -38, 38))
    roll = float(np.clip(rng.normal(0, 4), -12, 12))
    g = warp_3d(g, yaw, pitch, roll, rng.uniform(-0.25, 0.25), rng.uniform(2.0, 6.0))
    scale = rng.uniform(0.12, 1.0) * 100 / max(g.shape)                                # разное разрешение
    scale = min(scale * 3, 1.0)
    g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.5) * max(1.0, 1.0 / max(scale, 0.05)) * 0.6)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = (g > rng.uniform(70, 170)).astype(np.uint8) * 255
    m = D.tight(m)
    if m is None or m.shape[0] < 7 or m.shape[1] < 2:
        return None
    return D.feature(m)


def samples_from_fonts(per_glyph, rng):
    X, y = [], []
    fonts = [f for f in FONTS if os.path.exists(f)]
    for f in fonts:
        for d in range(10):
            glyph = render_glyph(str(d), f)
            for _ in range(per_glyph):
                feat = augment(glyph, rng)
                if feat is not None:
                    X.append(feat); y.append(d)
    return X, y, len(fonts)


def samples_from_real(glyphs, per_glyph, rng):
    """glyphs: список (маска цифры белая на чёрном, цифра 0-9) - вырезаны с настоящих табло."""
    X, y = [], []
    for mask, d in glyphs:
        for _ in range(per_glyph):
            feat = augment(mask, rng)
            if feat is not None:
                X.append(feat); y.append(d)
    return X, y


def train(X, y, out_path, seed=0):
    from sklearn.neural_network import MLPClassifier
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    mean, std = X.mean(0), X.std(0) + 1e-3
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=80, early_stopping=True,
                        n_iter_no_change=6, random_state=seed)
    clf.fit((X - mean) / std, y)
    arrays = {"n": np.array(len(clf.coefs_)), "mean": mean, "std": std}
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    np.savez_compressed(out_path, **arrays)
    return float(clf.best_validation_score_)

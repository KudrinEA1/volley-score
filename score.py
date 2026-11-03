"""Состояние счёта и проверки. Принимает показания распознавателя и решает, можно ли им верить.

Правила:
  * новое значение принимается, только если оно продержалось несколько кадров подряд;
  * очки меняются на +1 у одной стороны (откат на -1 - только после более долгого подтверждения);
  * после завершённой партии допускается сброс 0:0;
  * если счёт перелистнули сразу на 2-4 очка, он принимается после 3 кадров;
  * если пропущено больше очков (только вверх), принимается после 6 кадров подряд;
  * скачок, похожий на путаницу цифр 1 и 7 (31 -> 37), никогда не принимается сам: только ручная поправка;
  * сброс табло в 0:0 посреди партии принимается после 5 кадров подряд;
  * любое другое правдоподобное изменение, которое держится 10 кадров подряд (судья поправил счёт), принимается;
  * кадры, на которых табло не видно, подтверждение не сбрасывают (его сбрасывает только время);
  * скачок, который не подходит ни под одно правило и похож на ошибку чтения, отбрасывается;
  * партия засчитывается, когда завершённый счёт (например 25:10) продержался на табло несколько кадров;
    если табло откатили назад, партия снимается; при сбросе 0:0 она второй раз не засчитывается;
  * первое подтверждённое показание принимается как есть (запуск посреди матча);
  * ручная поправка на время "замораживает" распознавание, чтобы оно не спорило с оператором.
"""
import json
import os
import threading
import time

CONFIRM_UP = 2        # кадров подряд для +1
CONFIRM_DOWN = 4      # кадров подряд для -1
CONFIRM_SET = 3       # кадров подряд для новой партии и для первого показания
CONFIRM_JUMP = 3      # кадров подряд, если счёт перелистнули сразу на 2-4 очка
HEAL = 6              # кадров подряд, если пропущено больше (судья долго не листал)
JUMP_MAX_TOTAL = 4    # до скольких очков суммарно считается "несколько сразу"
BOARD_RESET = 5       # кадров подряд, если табло сбросили в 0:0 посреди партии
HEAL_ANY = 10         # кадров подряд для любого правдоподобного изменения (судья поправил счёт)
CAND_TTL = 12         # секунд: подтверждение, которому больше, считается устаревшим и начинается заново
MAX_POINTS = 45       # больше этого очков в партии не бывает: такое показание - ошибка чтения
FINISH_HOLD = 3       # кадров подряд, пока завершённый счёт держится на табло, чтобы засчитать партию
HOLD_POINTS = 15      # секунд тишины распознавания очков после ручной поправки
HOLD_SETS = 60        # то же для партий


class ScoreState:
    def __init__(self, set_points=25, last_set_points=15, sets_to_win=3, path=None):
        self.set_points = set_points
        self.last_set_points = last_set_points
        self.sets_to_win = sets_to_win
        self.path = path
        self.lock = threading.RLock()
        self.points = [0, 0]                   # по сторонам табло: [левая, правая]
        self.sets = [0, 0]
        self.swap = False                      # True: наши справа
        self.names = {"us": "Наши", "them": "Соперник"}
        self.points_ready = False              # было ли первое подтверждённое показание очков
        self.sets_ready = False                # то же для партий (табло их показывает)
        self._pc, self._pn = None, 0           # кандидат очков и счётчик подтверждений
        self._sc, self._sn = None, 0           # кандидат партий и счётчик
        self._pt = 0.0                         # когда кандидат очков видели в последний раз
        self._st = 0.0                         # то же для партий
        self._hold_p = 0.0
        self._hold_s = 0.0
        self._credited = False                 # партия уже засчитана по завершённому счёту
        self._credit_winner = None
        self._fin_n = 0                        # сколько кадров подряд держится завершённый счёт
        self.status = "ожидание кадров"
        self.updated = time.time()
        self.strict = False                    # режим матча: камера только продолжает трек, "сильно меньше" не принимается
        self.last_ok = time.time()             # когда камера в последний раз видела логичный счёт (для автопитстопа)
        self.load()

    # ---------- проверки ----------
    def _target(self):
        decider = self.sets[0] == self.sets[1] == self.sets_to_win - 1
        return self.last_set_points if decider else self.set_points

    def is_finished(self, p):
        return max(p) >= self._target() and abs(p[0] - p[1]) >= 2

    def _credit_finished(self, p):
        """Партия закончилась: засчитываем её лидеру."""
        w = 0 if p[0] > p[1] else 1
        self.sets[w] = min(self.sets_to_win, self.sets[w] + 1)
        self._credited, self._credit_winner = True, w

    def _uncredit(self):
        """Счёт откатили назад: партия ещё идёт, снимаем засчитанную."""
        if self._credited and self._credit_winner is not None:
            w = self._credit_winner
            self.sets[w] = max(0, self.sets[w] - 1)
        self._credited, self._credit_winner = False, None

    @staticmethod
    def _suspicious(cur, cand):
        """Показание отличается от текущего только путаницей 1 и 7 (31 -> 37, 11 -> 17): скорее всего, это ошибка чтения."""
        for a, b in zip(cur, cand):
            if a != b and str(a).replace("7", "1") == str(b).replace("7", "1"):
                return True
        return False

    # ---------- приём показаний ----------
    def update(self, reading, swap=None):
        with self.lock:
            if swap is not None:
                self.swap = bool(swap)
            self.updated = time.time()
            if not reading or not reading.get("mode"):
                self.status = "табло не видно, держу прошлый счёт"
                return
            lp, rp = reading["left"]["points"], reading["right"]["points"]
            if lp is None or rp is None:
                self.status = "цифры не читаются, держу прошлый счёт"
            elif time.time() >= self._hold_p:
                self._update_points((lp, rp))
            ls, rs = reading["left"]["sets"], reading["right"]["sets"]
            if ls is not None and rs is not None and time.time() >= self._hold_s:
                self._update_sets((ls, rs))
            self.save()

    def _count(self, kind, cand):
        attr_c, attr_n, attr_t = ("_pc", "_pn", "_pt") if kind == "p" else ("_sc", "_sn", "_st")
        now = time.time()
        if getattr(self, attr_c) == cand and now - getattr(self, attr_t) <= CAND_TTL:
            setattr(self, attr_n, getattr(self, attr_n) + 1)
        else:
            setattr(self, attr_c, cand)
            setattr(self, attr_n, 1)
        setattr(self, attr_t, now)
        return getattr(self, attr_n)

    def _update_points(self, p):
        cur = tuple(self.points)
        if self.points_ready and p == cur:
            self.last_ok = time.time()
            self._pc, self._pn = None, 0
            if not self.sets_ready and not self._credited and self.is_finished(cur):
                self._fin_n += 1
                if self._fin_n >= FINISH_HOLD:
                    self._credit_finished(cur)
                    self.status = "партия засчитана"
                    return
            else:
                self._fin_n = 0
            self.status = "ок"
            return
        n = self._count("p", p)
        if not self.points_ready:
            if n >= CONFIRM_SET:
                self.points, self.points_ready = list(p), True
                self._pc, self._pn = None, 0
                self._credited = self.is_finished(p) and not self.sets_ready   # запуск на уже завершённой партии: её не считаем
                self._credit_winner, self._fin_n = None, 0
                self.status = f"начальный счёт {p[0]}:{p[1]}"
            else:
                self.status = "проверяю первое показание"
            return
        d = (p[0] - cur[0], p[1] - cur[1])
        up = min(d) >= 0
        if d in ((1, 0), (0, 1)):
            need, kind = CONFIRM_UP, "очко"
        elif d in ((-1, 0), (0, -1)):
            need, kind = CONFIRM_DOWN, "откат"
        elif p == (0, 0) and self.is_finished(cur):
            need, kind = CONFIRM_SET, "новая партия"
        elif up and d[0] + d[1] <= JUMP_MAX_TOTAL:
            need, kind = CONFIRM_JUMP, "несколько очков сразу"
        elif up and max(p) <= MAX_POINTS and not self._suspicious(cur, p):
            need, kind = HEAL, "догоняю пропущенные очки"
        elif self.strict:
            self.status = f"не по ходу игры: {cur[0]}:{cur[1]} -> {p[0]}:{p[1]}"     # в матче такое принимается только вручную
            return
        elif max(p) <= 5 and max(cur) >= 12:
            need, kind = CONFIRM_SET + 1, "новая партия"        # финал партии мы не видели, но началась следующая
        elif p == (0, 0):
            need, kind = BOARD_RESET, "табло сброшено в 0:0"
        elif max(p) <= MAX_POINTS and not self._suspicious(cur, p):
            need, kind = HEAL_ANY, "принимаю новое показание"          # судья поправил счёт: если держится долго, это правда
        else:
            self.status = f"отброшено {cur[0]}:{cur[1]} -> {p[0]}:{p[1]}"
            return
        self.last_ok = time.time()
        if n >= need:
            if kind == "новая партия":
                if not self.sets_ready and not self._credited and max(cur) >= self._target() - 3:
                    self._credit_finished(cur)                 # табло партий не показывает: считаем сами
                self._credited, self._credit_winner = False, None
            elif self._credited and not self.is_finished(p):
                self._uncredit()                               # счёт откатили: партия ещё идёт
            self.points = list(p)
            self._fin_n = 0
            self._pc, self._pn = None, 0
            self.status = f"{kind}: {p[0]}:{p[1]}"
        else:
            self.status = f"проверяю: {kind}"

    def _update_sets(self, s):
        if max(s) > self.sets_to_win:
            return                                              # заведомо ошибка чтения
        cur = tuple(self.sets)
        if self.sets_ready and s == cur:
            self._sc, self._sn = None, 0
            return
        n = self._count("s", s)
        if not self.sets_ready and self.strict:
            # в матче партии известны: табло их только подтверждает (тот же счёт) или прибавляет одну партию
            d = (s[0] - cur[0], s[1] - cur[1])
            if s == cur or d in ((1, 0), (0, 1)):
                if n >= CONFIRM_SET:
                    self.sets, self.sets_ready = list(s), True
                    self._sc, self._sn = None, 0
            return
        if not self.sets_ready:
            if n >= CONFIRM_SET:
                self.sets, self.sets_ready = list(s), True
                self._sc, self._sn = None, 0
            return
        d = (s[0] - cur[0], s[1] - cur[1])
        if d in ((1, 0), (0, 1)):
            need = CONFIRM_SET
        elif d in ((-1, 0), (0, -1)):
            need = CONFIRM_DOWN
        elif s == (0, 0) and max(cur) == self.sets_to_win:
            need = CONFIRM_SET                                  # начался новый матч
        else:
            return
        if n >= need:
            self.sets = list(s)
            self._sc, self._sn = None, 0

    # ---------- ручные действия ----------
    def _idx(self, who):
        us = 1 if self.swap else 0
        return us if who == "us" else 1 - us

    def adjust(self, who, field, delta):
        with self.lock:
            i = self._idx(who)
            if field == "points":
                self.points[i] = max(0, self.points[i] + delta)
                self.points_ready = True
                self._pc, self._pn = None, 0
                self._hold_p = time.time() + HOLD_POINTS
                self._fin_n = 0
                if self._credited and not self.is_finished(tuple(self.points)):
                    self._uncredit()
            elif field == "sets":
                self.sets[i] = min(self.sets_to_win, max(0, self.sets[i] + delta))
                self._sc, self._sn = None, 0
                self._hold_s = time.time() + HOLD_SETS
                self._credited = self.is_finished(tuple(self.points))   # партии поправили вручную: сами их больше не трогаем
                self._credit_winner = None
            self.status = "ручная поправка"
            self.updated = time.time()
            self.save()

    def reset(self):
        with self.lock:
            self.points, self.sets = [0, 0], [0, 0]
            self.points_ready = self.sets_ready = False
            self._pc, self._pn, self._sc, self._sn = None, 0, None, 0
            self._pt = self._st = 0.0
            self._hold_p = self._hold_s = 0.0
            self._credited, self._credit_winner, self._fin_n = False, None, 0
            self.status = "сброс"
            self.save()

    def flip_sides(self):
        """Команды поменялись сторонами на табло: очки и партии переезжают вместе с командами."""
        with self.lock:
            self.swap = not self.swap
            self.points.reverse()
            self.sets.reverse()
            if self._credit_winner is not None:
                self._credit_winner = 1 - self._credit_winner
            self._pc, self._pn, self._sc, self._sn = None, 0, None, 0
            self.save()

    def set_team_values(self, us_pts, them_pts, us_sets, them_sets):
        """Питстоп: выставить счёт по командам (наши/соперник)."""
        with self.lock:
            u, t = self._idx("us"), self._idx("them")
            self.points[u], self.points[t] = int(us_pts), int(them_pts)
            self.sets[u], self.sets[t] = int(us_sets), int(them_sets)
            self.points_ready = True
            self._pc, self._pn, self._sc, self._sn, self._fin_n = None, 0, None, 0, 0
            self._credited, self._credit_winner = self.is_finished(tuple(self.points)), None
            self.last_ok = time.time()
            self.status = "счёт выставлен вручную"
            self.updated = time.time()
            self.save()

    def set_names(self, us, them):
        with self.lock:
            self.names = {"us": (us or "Наши")[:40], "them": (them or "Соперник")[:40]}
            self.save()

    # ---------- вывод ----------
    def snapshot(self):
        with self.lock:
            u, t = self._idx("us"), self._idx("them")
            return {
                "us": {"name": self.names["us"], "points": self.points[u], "sets": self.sets[u]},
                "them": {"name": self.names["them"], "points": self.points[t], "sets": self.sets[t]},
                "show_sets": self.sets_ready or any(self.sets),
                "status": self.status,
                "updated": self.updated,
            }

    # ---------- сохранение (переживает перезапуск сервера) ----------
    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"points": self.points, "sets": self.sets, "names": self.names, "swap": self.swap,
                           "points_ready": self.points_ready, "sets_ready": self.sets_ready,
                           "credited": self._credited, "credit_winner": self._credit_winner}, f, ensure_ascii=False)
        except OSError:
            pass

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.points, self.sets = d["points"], d["sets"]
            self.names, self.swap = d["names"], d["swap"]
            self.points_ready, self.sets_ready = d["points_ready"], d["sets_ready"]
            self._credited, self._credit_winner = d.get("credited", False), d.get("credit_winner")
        except (OSError, ValueError, KeyError):
            pass


# ====================== статистика очков по партиям ======================
# Записывает, кто взял каждое очко. Берёт счёт, который уже идёт на плашку, и дополнительно защищается от мусора
# (мяч попал в табло, карточки перелистывают, цифры на время случайные):
#   * изменение попадает в статистику, только если продержалось STAT_STABLE секунд без изменений;
#   * внутри партии очки только растут: +1..+4 (если обе команды выросли сразу - очки помечаются "порядок неизвестен");
#   * уменьшение на 1-3 (судья исправил) убирает последние очки этой команды;
#   * всё остальное (17:19 -> 3:2 посреди партии) принимается, только если держится STAT_ODD секунд;
#   * партия закрывается, когда после законно завершённого счёта (25 / 15 в пятой, разница 2, бывает и 30:28) табло показало 0:0;
#   * ручные кнопки пишутся сразу (immediate).
STAT_STABLE = 4.0
STAT_ODD = 10.0


class MatchStats:
    def __init__(self, path=None):
        self.path = path
        self.lock = threading.RLock()
        self.sets = []
        self.cur = [0, 0]                     # очки в статистике текущей партии [наши, соперник]
        self.cand, self.cand_t = None, 0.0
        self.closed = 0                       # сколько партий закрыто (номер последнего итога)
        self._new_set()
        self.load()

    # ---------- служебное ----------
    def _new_set(self, now=None):
        self.sets.append({"start": now or time.time(), "end": None, "seq": [], "final": None, "partial": False})
        self.cur = [0, 0]

    @staticmethod
    def _finished(p, target):
        return max(p) >= target and abs(p[0] - p[1]) >= 2

    def _kind(self, s, target):
        cur = tuple(self.cur)
        du, dt = s[0] - cur[0], s[1] - cur[1]
        if s == (0, 0) and self._finished(cur, target):
            return "newset"
        if du >= 0 and dt >= 0 and du + dt <= 4:
            return "up"
        if du <= 0 and dt <= 0 and -(du + dt) <= 3:
            return "down"
        return "odd"

    def _close(self, now, partial=False):
        st = self.sets[-1]
        st["final"], st["end"], st["partial"] = list(self.cur), now, partial
        self.closed += 1
        self._new_set(now)

    def _apply(self, s, target, now):
        kind = self._kind(s, target)
        if kind == "newset":
            self._close(now)
            self.save()
            return True
        if kind == "odd" and max(s) <= 5 and max(self.cur) >= 12:       # конец партии не увидели, но идёт уже следующая
            self._close(now, partial=True)
        seq = self.sets[-1]["seq"]
        d = [s[0] - self.cur[0], s[1] - self.cur[1]]
        for team in (0, 1):                                            # исправления вниз: убираем последние очки команды
            for _ in range(max(0, -d[team])):
                for k in range(len(seq) - 1, -1, -1):
                    if seq[k][0] == team:
                        del seq[k]
                        break
        both = d[0] > 0 and d[1] > 0
        for team in (0, 1):
            unsure = both or d[team] > 4                               # порядок очков неизвестен
            for _ in range(max(0, d[team])):
                seq.append([team, 1 if unsure else 0])
        self.cur = list(s)
        self.save()
        return True

    # ---------- приём счёта ----------
    def observe(self, score, target, immediate=False, now=None):
        """score: (наши, соперник) с плашки; target: до скольких очков идёт партия. True - статистика изменилась."""
        now = now or time.time()
        s = (int(score[0]), int(score[1]))
        with self.lock:
            if s == tuple(self.cur):
                self.cand = None
                return False
            if immediate:
                self.cand = None
                return self._apply(s, target, now)
            if s != self.cand:
                self.cand, self.cand_t = s, now
                return False
            need = STAT_ODD if self._kind(s, target) == "odd" else STAT_STABLE
            if now - self.cand_t >= need:
                self.cand = None
                return self._apply(s, target, now)
            return False

    def sync(self, played):
        """Номер партии в статистике всегда = сыграно партий + 1 (played = сумма партий по счёту).
        Партии выросли любым способом (кнопка, питстоп, камера) - текущая закрывается, пропущенные добавляются
        пустыми «нет данных»; уменьшились - лишние пустые партии убираются."""
        with self.lock:
            changed = False
            while len(self.sets) - 1 < played:
                now = time.time()
                st = self.sets[-1]
                if st["seq"] or self.cur != [0, 0]:
                    st["final"], st["end"] = list(self.cur), now
                    st["partial"] = st.get("partial", False) or not self._finished(tuple(self.cur), 25 if len(self.sets) < 5 else 15)
                else:
                    st.update(final=None, end=now, partial=True, nodata=True)
                self.closed += 1
                self._new_set(now)
                changed = True
            while len(self.sets) - 1 > played and len(self.sets) > 1:
                # партий стало меньше: сначала убираем пустые «нет данных» перед текущей партией (текущая сдвигается на их место)
                k = next((i for i in range(len(self.sets) - 2, -1, -1) if self.sets[i].get("nodata")), None)
                if k is not None:
                    del self.sets[k]
                elif not self.sets[-1]["seq"]:
                    self.sets.pop()                         # текущая пустая - возвращаемся в прошлую партию
                    last = self.sets[-1]
                    self.cur = list(last["final"]) if last.get("final") is not None else \
                        [sum(1 for t, _ in last["seq"] if t == 0), sum(1 for t, _ in last["seq"] if t == 1)]
                    last.update(final=None, end=None, nodata=False)
                else:
                    break                                   # в партиях есть записанные розыгрыши: не стираем
                self.closed = max(0, self.closed - 1)
                changed = True
            if changed:
                self.cand = None
                self.save()
            return changed

    # ---------- ручные действия ----------
    def close_set(self):
        with self.lock:
            if self.cur != [0, 0] or self.sets[-1]["seq"]:
                self._close(time.time(), partial=True)
                self.save()

    def reset_set(self):
        with self.lock:
            st = self.sets[-1]
            st["seq"], st["start"] = [], time.time()
            self.cur, self.cand = [0, 0], None
            self.save()

    def reset_match(self):
        with self.lock:
            self.sets, self.cand, self.closed = [], None, 0
            self._new_set()
            self.save()

    def dump(self):
        with self.lock:
            return json.loads(json.dumps({"sets": self.sets, "cur": self.cur, "closed": self.closed}))

    def restore(self, d):
        with self.lock:
            self.sets, self.cur, self.closed = d["sets"], d["cur"], d["closed"]
            self.cand = None
            self.save()

    # ---------- вывод ----------
    def streak(self):
        """(команда 0/1, сколько очков подряд) по текущей партии; очки с неизвестным порядком серию обрывают."""
        with self.lock:
            seq = self.sets[-1]["seq"]
            if not seq or seq[-1][1]:
                return None, 0
            team, n = seq[-1][0], 0
            for t, q in reversed(seq):
                if t != team or q:
                    break
                n += 1
            return team, n

    def last_closed(self):
        with self.lock:
            done = [st for st in self.sets if st["final"] is not None]
            return (len(done), done[-1]) if done else (0, None)

    def table(self):
        with self.lock:
            return [{"n": i + 1, "final": st["final"] if st["final"] is not None else list(self.cur) if i == len(self.sets) - 1 else None,
                     "seq": st["seq"], "done": st["final"] is not None, "partial": st["partial"], "nodata": bool(st.get("nodata")),
                     "minutes": round(((st["end"] or time.time()) - st["start"]) / 60)}
                    for i, st in enumerate(self.sets) if st["final"] is not None or st["seq"] or st.get("nodata") or i == len(self.sets) - 1]

    # ---------- сохранение ----------
    def save(self):
        if not self.path:
            return
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"sets": self.sets, "cur": self.cur, "closed": self.closed}, f)
        except OSError:
            pass

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.sets, self.cur, self.closed = d["sets"], d["cur"], d.get("closed", 0)
            if not self.sets:
                self._new_set()
        except (OSError, ValueError, KeyError):
            pass

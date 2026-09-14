"""四层记忆系统（harness 用）。

  L1 Working    会话内短期上下文：当前对话 messages / 任务临时状态（不落盘）
  L2 Episodic   情景记忆：历史任务记录（跨会话），按相关性 top-k 召回供参考
  L3 Semantic   语义/长期记忆：项目事实、领域知识、用户约定（注入 system）
  L4 Procedural 程序/技能记忆：skill 渐进披露展开状态、调用统计（防重复披露/死循环）

落盘目录：<store_dir>/（默认 agent/data/memory/）：
  episodic.jsonl / semantic.json / procedural.json
"""
from __future__ import annotations

import json
import os
import time


def _overlap_score(q: str, s: str) -> int:
    """极简相关性：共现查询词数量（零依赖）。"""
    qs = {w for w in q if w.strip()} | set(q.split())
    if not qs:
        return 0
    body = s or ""
    return sum(1 for w in qs if w in body)


class MemoryManager:
    def __init__(self, store_dir: str):
        self.dir = store_dir
        os.makedirs(self.dir, exist_ok=True)
        self._ep_path = os.path.join(self.dir, "episodic.jsonl")
        self._sem_path = os.path.join(self.dir, "semantic.json")
        self._pro_path = os.path.join(self.dir, "procedural.json")
        # L1 working（仅内存）
        self.working: dict = {}
        # L2 episodic
        self.episodic: list[dict] = []
        self._load_episodic()
        # L3 semantic
        self.semantic: dict[str, str] = self._load_json(self._sem_path, {})
        # L4 procedural
        self.procedural: dict = self._load_json(self._pro_path, {"skills": {}})
        if "skills" not in self.procedural:
            self.procedural["skills"] = {}

    # ---------- 持久化 helpers ----------
    def _load_json(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _load_episodic(self):
        if os.path.exists(self._ep_path):
            with open(self._ep_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.episodic.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass

    def save(self):
        """三层记忆落盘：episodic/semantic/procedural 全部 fsync 强制写入。"""
        with open(self._ep_path, "w", encoding="utf-8") as f:
            for e in self.episodic[-200:]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with open(self._sem_path, "w", encoding="utf-8") as f:
            json.dump(self.semantic, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(self._pro_path, "w", encoding="utf-8") as f:
            json.dump(self.procedural, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    def flush(self) -> dict:
        """主动 memory flush：强制落盘 + 返回各层摘要（供 UI 反馈）。

        返回：
          episodic_count / semantic_count / procedural_skill_count
          last_episodic_query / last_episodic_ts
          files: {path, size, mtime}
        """
        self.save()
        files = {}
        for name, p in (("episodic", self._ep_path),
                        ("semantic", self._sem_path),
                        ("procedural", self._pro_path)):
            try:
                st = os.stat(p)
                files[name] = {"path": p, "size": st.st_size,
                                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}
            except Exception:
                files[name] = {"path": p, "size": 0, "mtime": "?"}
        last = self.episodic[-1] if self.episodic else {}
        return {
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "procedural_skill_count": len(self.procedural.get("skills", {}) or {}),
            "last_episodic_query": (last.get("query", "") or "")[:60],
            "last_episodic_ts": last.get("ts", "—"),
            "files": files,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ---------- L1 Working ----------
    def working_set(self, key, value):
        self.working[key] = value

    def working_get(self, key, default=None):
        return self.working.get(key, default)

    # ---------- L2 Episodic ----------
    def episodic_add(self, query: str, outcome: str, meta: dict = None):
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "outcome": outcome,
            "meta": meta or {},
        }
        self.episodic.append(rec)
        self.save()

    def episodic_search(self, query: str, k: int = 3) -> list[dict]:
        scored = []
        for e in self.episodic:
            hay = (e.get("query", "") + " " + str(e.get("outcome", "")))
            sc = _overlap_score(query, hay)
            if sc > 0:
                scored.append((sc, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:k]]

    # ---------- L3 Semantic ----------
    def semantic_set(self, key: str, value: str):
        self.semantic[key] = value
        self.save()

    def semantic_get(self, key: str, default: str = "") -> str:
        return self.semantic.get(key, default)

    # ---------- L4 Procedural（技能记忆）----------
    def skill_expanded(self, name: str) -> bool:
        return bool(self.procedural["skills"].get(name, {}).get("expanded"))

    def skill_mark_expanded(self, name: str):
        self.procedural["skills"].setdefault(name, {})["expanded"] = True
        self.save()

    def skill_note_call(self, name: str):
        s = self.procedural["skills"].setdefault(name, {})
        s["calls"] = s.get("calls", 0) + 1
        s["last"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def skill_calls(self, name: str) -> int:
        return int(self.procedural["skills"].get(name, {}).get("calls", 0))

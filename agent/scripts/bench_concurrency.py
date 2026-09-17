"""实测：不同并发下 review 任务的内存峰值与耗时（用于定队列/并发/令牌桶参数）。"""
import os
import sys
import time
import threading
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import psutil  # noqa: E402

PROC = psutil.Process()
BASE_MB = PROC.memory_info().rss / 1024 / 1024
peak_mb = [BASE_MB]
_stop = threading.Event()


def sampler():
    while not _stop.is_set():
        try:
            m = PROC.memory_info().rss / 1024 / 1024
            if m > peak_mb[0]:
                peak_mb[0] = m
        except Exception:
            pass
        time.sleep(0.05)


def build_agent():
    from subagents.review_agent import ReviewAgent
    return ReviewAgent()


def load_texts(n=6):
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "bench", "data", "real_fragments3")
    files = sorted(os.listdir(d))[:n]
    return [open(os.path.join(d, f), encoding="utf-8").read() for f in files]


def run_one(agent, text, q):
    return agent.run(query=q, plan=text,
                     ctx={"rag_rerank": True, "ner_crf": True, "memory_dir": None})


def measure(n_conc, texts):
    agent = build_agent()          # 预热（模型加载一次，模拟服务常驻）
    q = "全面审查该方案，重点核查危大工程判定与技术参数合规性"
    tasks = [texts[i % len(texts)] for i in range(n_conc)]
    peak_mb[0] = PROC.memory_info().rss / 1024 / 1024
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n_conc) as ex:
        list(ex.map(lambda t: run_one(agent, t, q), tasks))
    dt = time.perf_counter() - t0
    return dt, peak_mb[0]


def main():
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    texts = load_texts()
    print(f"基线内存 {BASE_MB:.0f} MB ｜ 样本 {len(texts)} 个真实片段")
    print(f"{'并发':>4} {'总耗时s':>9} {'单任务均s':>10} {'峰值内存MB':>11} {'增量MB':>9}")
    for n in (1, 2, 3):
        dt, peak = measure(n, texts)
        print(f"{n:>4} {dt:>9.2f} {dt/n:>10.2f} {peak:>11.0f} {peak-BASE_MB:>9.0f}")
    _stop.set()


if __name__ == "__main__":
    main()

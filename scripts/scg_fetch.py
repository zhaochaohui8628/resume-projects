# -*- coding: utf-8 -*-
"""上海建工标准文库 sms.scg.cn 批量抓取脚本。

复用浏览器登录态（Cookie），拉取标准列表并下载 PDF 到 data/raw/ 对应目录。

用法：
  export SCG_COOKIE="ASP.NET_SessionId=...; HXLING-BPM-COOKIE-NAME=..."
  python scripts/scg_fetch.py --category 824          # 全文强制性规范(GB550xx) 列清单
  python scripts/scg_fetch.py --category 824 --download # 实际下载 PDF
  python scripts/scg_fetch.py --search "JGJ 120" --download  # 按编号搜索下载

分类 code(NavTag)：
  824=全文强制性规范  821=行业标准  820=国家标准  172292=上海地标-标准
  172293=上海地标-图集  823=团体标准  -12=企业标准
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

BASE = "https://sms.scg.cn"
LIST_URL = BASE + "/PDCA/ashx/CommentStandardHandler.ashx"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

# 分类 code -> (treeChildrens 叶子 NavTag, 输出子目录)
CATEGORY_MAP = {
    "824": ("824", "standards"),        # 全文强制性规范
    "821": ("821", "standards"),        # 行业标准
    "820": ("824,-20", "standards"),    # 国家标准
    "172292": ("172292", "guidelines"),  # 上海地标-标准
    "172293": ("172293", "guidelines"),  # 上海地标-图集
    "823": ("823", "guidelines"),       # 团体标准
    "-12": ("-12", "guidelines"),       # 企业标准
}


def guess_category(kw: str) -> str:
    up = kw.upper().replace(" ", "")
    if up.startswith("DG") or up.startswith("DBJ") or up.startswith("DB"):
        return "172292"          # 上海地方标准
    if up.startswith("JGJ") or up.startswith("JG/") or up.startswith("JG") or up.startswith("CJJ"):
        return "821"             # 行业标准
    if up.startswith("GB"):
        return "820"             # 国家标准
    return "821"


def search_one(code, tree_childrens, kw):
    """在指定分类按编号搜索，返回命中的第一行（按编号精确优先）。"""
    rows, _ = fetch_list(code, tree_childrens, search_value=kw)
    if not rows:
        return None
    norm = kw.upper().replace(" ", "").replace("/", "")
    for r in rows:
        no = r.get("JSON_std_no", "").upper().replace(" ", "").replace("/", "")
        if no == norm or no.startswith(norm):
            return r
    return rows[0]


def _http(url, data=None, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={
                "User-Agent": UA,
                "Cookie": os.environ.get("SCG_COOKIE", ""),
                "Referer": BASE + "/PDCA/ComStandardSearch.aspx",
                "X-Requested-With": "XMLHttpRequest",
            })
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5)
    raise last


def fetch_list(code, tree_childrens, search_value="", page=1, rows=100):
    q = urllib.parse.urlencode({"action": "query", "code": code, "n": int(time.time() * 1000)})
    body = urllib.parse.urlencode({
        "action": "query", "searchValue": search_value, "type": 0,
        "treeChildrens": tree_childrens, "isbuy": 0, "IsQt": 0,
        "isstruct": 0, "isfile": 0, "page": page, "rows": rows,
    }).encode()
    data = json.loads(_http(LIST_URL + "?" + q, data=body).decode("utf-8"))
    return data.get("rows", []), int(data.get("total", 0))


def iter_category(code, tree_childrens, search_value=""):
    page, total = 1, None
    while True:
        rows, total = fetch_list(code, tree_childrens, search_value, page)
        if not rows:
            break
        yield from rows
        if len(rows) < 100 or page * 100 >= total:
            break
        page += 1


def norm_name(std_no, std_name):
    no = re.sub(r"\s+", "", std_no).replace("/", "-")
    name = re.sub(r"[\\/:*?\"<>|\s]+", "", std_name)
    return f"{no}_{name}.pdf"


def download_pdf(std_file, out_dir, std_no, std_name):
    if not std_file or not std_file.lower().endswith(".pdf"):
        return False, "no-pdf"
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, norm_name(std_no, std_name))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True, "exists"
    for sub in ("basicStandard", "coStandard"):
        url = f"{BASE}/assets/{sub}/{std_file}"
        try:
            data = _http(url)
            if data[:4] == b"%PDF":
                with open(dst, "wb") as f:
                    f.write(data)
                return True, f"{sub}:{len(data)}B"
        except Exception as e:
            continue
    return False, "404"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", help="分类 code")
    ap.add_argument("--search", help="按编号/名称搜索")
    ap.add_argument("--file", help="编号清单文件（每行：编号 或 编号|分类code，# 注释）")
    ap.add_argument("--download", action="store_true", help="实际下载 PDF（否则只列清单）")
    ap.add_argument("--outdir", help="输出子目录（默认按分类映射）")
    args = ap.parse_args()

    if not os.environ.get("SCG_COOKIE"):
        print("[error] 请先 export SCG_COOKIE=...", file=sys.stderr)
        sys.exit(1)

    entries = []  # [(row, 输出子目录)]
    if args.category:
        tc, out_sub = CATEGORY_MAP.get(args.category, (args.category, "standards"))
        entries = [(r, out_sub) for r in iter_category(args.category, tc)]
        print(f"[category {args.category}] 共 {len(entries)} 本")
    elif args.file:
        lines = [l.strip() for l in open(args.file, encoding="utf-8")
                 if l.strip() and not l.startswith("#")]
        for line in lines:
            parts = line.split("|")
            kw = parts[0].strip()
            code = parts[1].strip() if len(parts) > 1 else guess_category(kw)
            tc, sub = CATEGORY_MAP.get(code, (code, "standards"))
            hit = search_one(code, tc, kw)
            if hit:
                entries.append((hit, sub))
                print(f"  [命中] {hit.get('JSON_std_no')} {hit.get('JSON_std_name')} -> {sub}/")
            else:
                print(f"  [未命中] {kw} (分类 {code})")
    elif args.search:
        for code in ("821", "820", "172292"):
            tc, sub = CATEGORY_MAP.get(code, (code, "standards"))
            for r in iter_category(code, tc, search_value=args.search):
                entries.append((r, sub))
        # 去重
        seen, uniq = set(), []
        for r, sub in entries:
            k = r.get("JSON_std_no", "")
            if k and k not in seen:
                seen.add(k)
                uniq.append((r, sub))
        entries = uniq
        print(f"[search {args.search}] 命中 {len(entries)} 本")
    else:
        ap.print_help()
        sys.exit(0)

    ok = skip = fail = nofile = 0
    for r, sub in entries:
        no = r.get("JSON_std_no", "").strip()
        name = r.get("JSON_std_name", "").strip()
        f = r.get("JSON_std_file", "").strip()
        out_dir = os.path.join(RAW, args.outdir or sub)
        if args.download:
            status, info = download_pdf(f, out_dir, no, name)
            if status:
                ok += 1
            elif info == "no-pdf":
                nofile += 1
            else:
                fail += 1
                print(f"  [fail] {no} {name} ({info})")
        else:
            print(f"  {no}  {name}  file={f or '(无文件)'}")

    if args.download:
        print(f"\n[完成] 成功 {ok} / 跳过(已存在) {skip} / 无文件 {nofile} / 失败 {fail}")
        print(f"输出目录：{os.path.join(RAW, args.outdir) if args.outdir else RAW}")


if __name__ == "__main__":
    main()

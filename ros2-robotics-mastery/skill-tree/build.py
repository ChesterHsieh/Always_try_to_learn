#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從 tree_data.py 產生 skill-tree.json 與 index.html

用法:  python3 build.py
"""
import json, pathlib, sys, collections

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from tree_data import META, N  # noqa: E402


def validate(nodes):
    ids = [n["id"] for n in nodes]
    errs = []
    dup = [i for i, c in collections.Counter(ids).items() if c > 1]
    if dup:
        errs.append(f"重複 id: {dup}")
    idset = set(ids)
    for n in nodes:
        for d in n["deps"]:
            if d not in idset:
                errs.append(f"{n['id']} 的前置 {d} 不存在")
        if n["act"] not in {a["id"] for a in META["acts"]}:
            errs.append(f"{n['id']} 的 act {n['act']} 不存在")
        if not n["tasks"]:
            errs.append(f"{n['id']} 沒有任務")
    # 循環偵測
    seen, stack = set(), set()

    def dfs(i):
        if i in stack:
            errs.append(f"依賴循環: {i}")
            return
        if i in seen:
            return
        stack.add(i); seen.add(i)
        for d in nodes_by[i]["deps"]:
            if d in nodes_by:
                dfs(d)
        stack.discard(i)

    nodes_by = {n["id"]: n for n in nodes}
    for i in ids:
        dfs(i)
    return errs


def stats(nodes):
    total = sum(n["hours"] for n in nodes)
    main = sum(n["hours"] for n in nodes if n["track"] == "main")
    side = total - main
    per_act = {}
    for n in nodes:
        per_act.setdefault(n["act"], {"n": 0, "h": 0})
        per_act[n["act"]]["n"] += 1
        per_act[n["act"]]["h"] += n["hours"]
    return total, main, side, per_act


def main():
    nodes = N
    errs = validate(nodes)
    if errs:
        print("❌ 驗證失敗:")
        for e in errs:
            print("  -", e)
        sys.exit(1)

    total, main_h, side_h, per_act = stats(nodes)
    meta = dict(META)
    meta["total_hours"] = total
    meta["main_hours"] = main_h
    meta["side_hours"] = side_h
    meta["node_count"] = len(nodes)

    payload = {"meta": meta, "nodes": nodes}

    (HERE / "skill-tree.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    tpl = (HERE / "index.template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__TREE_JSON__*/",
                       json.dumps(payload, ensure_ascii=False))
    (HERE / "index.html").write_text(html, encoding="utf-8")

    print(f"✅ 產生完成")
    print(f"   節點數    : {len(nodes)}")
    print(f"   總時數    : {total} h  (主線 {main_h} h / 支線 {side_h} h)")
    print(f"   主線週數  : {main_h / meta['weekly_hours']:.0f} 週 @ {meta['weekly_hours']}h/週")
    print(f"   全樹週數  : {total / meta['weekly_hours']:.0f} 週")
    print()
    for a in META["acts"]:
        s = per_act.get(a["id"], {"n": 0, "h": 0})
        print(f"   {a['id']:<4} {a['name']:<18} {s['n']:>2} 節點  {s['h']:>3} h")


if __name__ == "__main__":
    main()

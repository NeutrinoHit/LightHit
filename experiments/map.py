#!/usr/bin/env python
"""Write the idea map: one self-contained HTML page of the registry.

    python experiments/map.py                    # -> experiments/map.html
    python experiments/map.py --out PATH [--fragment]

The page reads only ``ideas/*.toml`` (with their generated ``[result]`` blocks)
and ``plain_ru.toml`` (short Russian names and plain outcomes). It shows the
registry as a tree (area -> status -> idea), the chains of superseded ideas and
a sortable table; a click on an idea opens its question, verdict, measured cases
and how to rerun them. ``--fragment`` omits the document skeleton (doctype,
html, head and body tags) for hosts that add their own.
"""

from __future__ import annotations

import argparse
import glob
import json
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent

CASE_KEYS = ("status", "date", "threads", "reference", "error", "wall_vs_baseline",
             "order_0", "order_1", "order_ge2", "order_total", "active_agreement")


def load_ideas() -> list[dict]:
    plain = tomllib.loads((HERE / "plain_ru.toml").read_text())
    ideas = []
    for path in sorted(glob.glob(str(HERE / "ideas" / "*.toml"))):
        d = tomllib.load(open(path, "rb"))
        ru = plain.get(d["id"], {})
        cases = []
        for name, r in d.get("result", {}).items():
            case = {k: r[k] for k in CASE_KEYS if k in r}
            case["name"] = name
            timing = r.get("timing", {})
            if "wall_s" in timing:
                case["wall_s"] = timing["wall_s"]
            if "peak_rss_mib" in timing:
                case["rss_mib"] = timing["peak_rss_mib"]
            metrics = r.get("metrics")
            if isinstance(metrics, dict):
                case["metrics"] = {k: v for k, v in list(metrics.items())[:12]}
            cases.append(case)
        code = d.get("code", {})
        ideas.append(dict(
            id=d["id"], title=d["title"], area=d["area"], kind=d["kind"],
            status=d["status"], verdict=d.get("verdict", ""),
            question=" ".join(d.get("question", "").split()),
            baseline=d.get("baseline", ""), run=d.get("how-to-run", ""),
            reason=d.get("not_reproducible_reason", ""),
            parts=d.get("not_reproducible_parts", []),
            name=ru.get("name", d["title"]), plain=ru.get("plain", ""),
            modules=code.get("modules", []), scripts=code.get("scripts", []),
            tests=code.get("tests", []), options=[" ".join(o.split()) for o in code.get("options", [])],
            cases=cases))
    return ideas


def render(fragment: bool) -> str:
    ideas = load_ideas()
    dates = sorted(c["date"][:10] for i in ideas for c in i["cases"] if "date" in c)
    meta = dict(ideas=len(ideas), cases=sum(len(i["cases"]) for i in ideas),
                failed=sum(c.get("status") != "ok" for i in ideas for c in i["cases"]),
                first=dates[0] if dates else "", last=dates[-1] if dates else "")
    data = json.dumps(dict(meta=meta, ideas=ideas), ensure_ascii=False).replace("</", "<\\/")
    page = PAGE.replace("__DATA__", data)
    if fragment:
        return page
    return ("<!doctype html>\n<html lang=\"ru\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")


PAGE = r"""<title>Реестр идей LightHit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Golos+Text:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Unbounded:wght@500&display=swap">
<style>
/* Layout: a tree per detector area (area node -> status branches -> idea chips),
   a drawer with the idea's record; Baikal ice neutrals, Cherenkov-blue accent. */
:root {
  --bg: #eef3f6; --surface: #ffffff; --raised: #f7fafc; --ink: #13202a; --muted: #586a77;
  --line: #c9d5dd; --accent: #1f5fbf; --accent-soft: #e2ecfa;
  --s-default: #17794a; --s-used: #3a8f68; --s-open: #a86b0f; --s-tool: #2d6aa3;
  --s-sup: #77838d; --s-rej: #bf3b2b; --s-nr: #7449ab; --bad: #bf3b2b;
  --f-display: "Unbounded", "Golos Text", system-ui, sans-serif;
  --f-body: "Golos Text", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --r: 8px;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0c141a; --surface: #141f28; --raised: #18252f; --ink: #e2e9ee; --muted: #93a3af;
  --line: #2b3b47; --accent: #79a8ff; --accent-soft: #1a2c47;
  --s-default: #4cc489; --s-used: #6fb894; --s-open: #e0a443; --s-tool: #6aa7e0;
  --s-sup: #8b97a1; --s-rej: #f07a68; --s-nr: #ad8ae6; --bad: #f07a68; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #0c141a; --surface: #141f28; --raised: #18252f; --ink: #e2e9ee; --muted: #93a3af;
  --line: #2b3b47; --accent: #79a8ff; --accent-soft: #1a2c47;
  --s-default: #4cc489; --s-used: #6fb894; --s-open: #e0a443; --s-tool: #6aa7e0;
  --s-sup: #8b97a1; --s-rej: #f07a68; --s-nr: #ad8ae6; --bad: #f07a68; color-scheme: dark }
* { box-sizing: border-box }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 var(--f-body) }
.wrap { max-width: 1240px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px }
h1 { font: 500 clamp(22px, 3.4vw, 32px)/1.15 var(--f-display); margin: 0 0 8px; letter-spacing: -0.01em; text-wrap: balance }
.lede { color: var(--muted); max-width: 68ch; margin: 0 }
.facts { display: flex; flex-wrap: wrap; gap: 6px 18px; margin-top: 10px; color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums }
.facts b { color: var(--ink); font-weight: 600 }
.legend { margin-top: 22px; display: grid; gap: 10px }
.bar { display: flex; height: 14px; border-radius: 7px; overflow: hidden; background: var(--line) }
.bar span { display: block; height: 100% }
.chips { display: flex; flex-wrap: wrap; gap: 8px }
.fchip { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); background: var(--surface);
  color: var(--ink); border-radius: 999px; padding: 5px 12px 5px 10px; font: inherit; font-size: 13px; cursor: pointer }
.fchip[aria-pressed="false"] { opacity: .45 }
.fchip:focus-visible, .leaf:focus-visible, .tab:focus-visible, th button:focus-visible, .close:focus-visible, .copy:focus-visible, .lnk:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; background: var(--c) }
.fchip .n { color: var(--muted); font-variant-numeric: tabular-nums }
.hint { color: var(--muted); font-size: 13px; margin: 0 }
.toolbar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin-top: 26px;
  position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); padding-block: 10px; border-bottom: 1px solid var(--line) }
.tabs { display: flex; gap: 4px; background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 3px }
.tab { border: 0; background: transparent; color: var(--muted); font: inherit; font-size: 14px; padding: 6px 14px; border-radius: 999px; cursor: pointer }
.tab[aria-selected="true"] { background: var(--accent); color: var(--surface) }
#q { font: inherit; font-size: 14px; padding: 7px 12px; border: 1px solid var(--line); border-radius: 999px; background: var(--surface);
  color: var(--ink); width: min(320px, 100%) }
.view { margin-top: 22px }
/* tree */
.area { display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 0 26px; padding-block: 18px; border-bottom: 1px solid var(--line) }
.area:last-child { border-bottom: 0 }
.anode { align-self: start; position: sticky; top: 76px }
.anode h2 { font: 500 17px/1.2 var(--f-display); margin: 0 0 4px }
.anode .cnt { font: 500 12px var(--f-mono); color: var(--muted) }
.anode p { margin: 6px 0 0; color: var(--muted); font-size: 13px }
.branches { position: relative; border-left: 2px solid var(--line); padding-left: 22px; display: grid; gap: 14px; min-width: 0 }
.branch { position: relative; min-width: 0 }
.branch::before { content: ""; position: absolute; left: -24px; top: 11px; width: 16px; border-top: 2px solid var(--line) }
.blabel { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; margin-bottom: 7px }
.blabel .n { color: var(--muted); font-weight: 400; font-variant-numeric: tabular-nums }
.leaves { display: flex; flex-wrap: wrap; gap: 7px }
.leaf { display: inline-flex; align-items: center; gap: 7px; max-width: 100%; text-align: left; border: 1px solid var(--line);
  background: var(--surface); color: var(--ink); border-radius: 6px; padding: 5px 10px 5px 8px; font: inherit; font-size: 13.5px; cursor: pointer }
.leaf:hover { border-color: var(--c) }
.leaf.main { border-color: var(--c); box-shadow: inset 3px 0 0 var(--c) }
.leaf .x { color: var(--bad); font: 500 11px var(--f-mono) }
.leaf span.t { overflow-wrap: anywhere }
/* evolution */
.evo { display: grid; gap: 14px }
.chain { display: grid; grid-template-columns: minmax(0, 1fr) 40px minmax(0, 260px); align-items: center; gap: 10px;
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--r); padding: 14px }
.olds { display: grid; gap: 8px; min-width: 0 }
.old { display: grid; gap: 2px; min-width: 0 }
.old .p { color: var(--muted); font-size: 13px }
.arrow { color: var(--muted); font-size: 22px; text-align: center }
.lnk { border: 0; background: none; padding: 0; color: var(--ink); font: inherit; font-weight: 600; text-align: left; cursor: pointer;
  text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 3px }
.lnk:hover { text-decoration-color: var(--accent) }
.new { display: grid; gap: 4px; padding: 10px 12px; border-radius: 6px; background: var(--accent-soft) }
.new small { color: var(--muted) }
/* table */
.tablebox { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--r); background: var(--surface) }
table { border-collapse: collapse; width: 100%; font-size: 13.5px }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top }
th { background: var(--raised); font-weight: 600; white-space: nowrap }
th button { border: 0; background: none; font: inherit; color: inherit; cursor: pointer; padding: 0 }
td.num { font-variant-numeric: tabular-nums; white-space: nowrap }
tr:last-child td { border-bottom: 0 }
.st { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap }
/* drawer */
.scrim { position: fixed; inset: 0; background: rgb(0 0 0 / .28); z-index: 20 }
aside { position: fixed; top: 0; right: 0; bottom: 0; width: min(640px, 100%); background: var(--surface); z-index: 21;
  border-left: 1px solid var(--line); overflow-y: auto; padding: calc(18px + env(safe-area-inset-top, 0px)) 24px calc(32px + env(safe-area-inset-bottom, 0px)) }
.dhead { display: flex; gap: 12px; align-items: flex-start; justify-content: space-between }
aside h3 { font: 500 20px/1.25 var(--f-display); margin: 0; text-wrap: balance }
.close { border: 1px solid var(--line); background: var(--raised); color: var(--ink); border-radius: 999px; width: 34px; height: 34px; font-size: 18px; cursor: pointer; flex: none }
.sub { color: var(--muted); font-size: 13px; margin: 4px 0 0 }
.tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px }
.tag { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; padding: 3px 9px; border-radius: 999px; background: var(--raised); border: 1px solid var(--line) }
.plain { margin: 16px 0 0; font-size: 16px; line-height: 1.5; padding: 12px 14px; border-radius: 6px; background: var(--accent-soft) }
aside h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 22px 0 6px; font-weight: 600 }
aside p { margin: 0 }
.en { font-size: 14px }
.cases { display: grid; gap: 8px }
.case { border: 1px solid var(--line); border-radius: 6px; padding: 9px 11px; min-width: 0 }
.case.err { border-color: var(--bad) }
.chead { display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: baseline; font-size: 13px }
.chead code { font: 500 12.5px var(--f-mono); overflow-wrap: anywhere }
.chead .m { color: var(--muted); font-variant-numeric: tabular-nums }
.errtext { color: var(--bad); font-size: 13px; margin-top: 4px; overflow-wrap: anywhere }
.orders { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 6px; margin-top: 7px }
.ord { background: var(--raised); border-radius: 5px; padding: 5px 8px; font-size: 12px; font-variant-numeric: tabular-nums }
.ord b { display: block; font-weight: 600; font-size: 11.5px; color: var(--muted) }
details { margin-top: 6px; font-size: 12.5px }
summary { cursor: pointer; color: var(--muted) }
.kv { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 2px 12px; margin-top: 4px; font: 12px var(--f-mono) }
.kv span { overflow-wrap: anywhere }
pre { margin: 0; padding: 10px 12px; background: var(--raised); border: 1px solid var(--line); border-radius: 6px; overflow-x: auto;
  font: 12.5px/1.5 var(--f-mono); white-space: pre-wrap; overflow-wrap: anywhere }
.runrow { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: start }
.copy { border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 6px; padding: 6px 10px; font: inherit; font-size: 12.5px; cursor: pointer }
ul.code { margin: 0; padding-left: 18px; font: 12.5px/1.6 var(--f-mono) }
ul.code li { overflow-wrap: anywhere }
.empty { color: var(--muted); padding: 30px 0 }
@media (max-width: 760px) {
  .area { grid-template-columns: 1fr; gap: 12px }
  .anode { position: static }
  .chain { grid-template-columns: 1fr }
  .arrow { transform: rotate(90deg); text-align: left; width: 24px }
  .toolbar { position: static }
  aside { padding-inline: 16px }
}
@media (prefers-reduced-motion: no-preference) { aside { animation: slide .18s ease-out } @keyframes slide { from { transform: translateX(24px); opacity: .6 } } }
</style>

<div class="wrap">
  <header>
    <h1>Реестр идей LightHit</h1>
    <p class="lede">Каждая идея, которую пробовали в LightHit: маршрут, настройка, проба, эталон или сборщик таблиц. Цвет показывает её судьбу: работает ли она в текущем расчёте, заменена ли лучшим решением, отвергнута или ждёт продолжения. Нажмите на идею, чтобы увидеть вопрос, итог, замеры и команду для повторного запуска.</p>
    <div class="facts" id="facts"></div>
  </header>
  <section class="legend" aria-label="Статусы">
    <div class="bar" id="bar" role="img"></div>
    <div class="chips" id="filters"></div>
    <p class="hint">Нажмите на статус, чтобы скрыть или показать его идеи.</p>
  </section>
  <div class="toolbar">
    <div class="tabs" role="tablist">
      <button class="tab" role="tab" id="tab-tree" aria-selected="true">Дерево</button>
      <button class="tab" role="tab" id="tab-evo" aria-selected="false">Что чем заменено</button>
      <button class="tab" role="tab" id="tab-table" aria-selected="false">Таблица</button>
    </div>
    <input id="q" type="search" placeholder="Поиск: название, модуль, скрипт" aria-label="Поиск">
  </div>
  <main class="view" id="view"></main>
</div>
<div class="scrim" id="scrim" hidden></div>
<aside id="drawer" hidden aria-modal="true" role="dialog" aria-labelledby="d-title"></aside>

<script>
const DATA = __DATA__;
const AREAS = [
  ["routes", "Маршруты событий", "Готовые расчёты целого события: ливень, мюон, событие, лазер, а также прежние варианты и эталоны."],
  ["shower", "Ливни", "Настройки и варианты расчёта света электромагнитных и адронных ливней."],
  ["muon", "Мюоны", "Первичный мюон, его вторичные ливни, таблицы для мюона и отбор датчиков."],
  ["core", "Ядро переноса", "Решатель уравнения переноса, ядро датчика, независимые эталоны и общие проверки."],
  ["tables", "Таблицы", "Готовые таблицы набора данных."],
  ["tools", "Инструменты", "Вспомогательные сборки."]];
const STATUS = [
  ["default", "По умолчанию", "s-default", "Это и есть текущий расчёт или его настройка по умолчанию."],
  ["used-by-default", "Часть расчёта", "s-used", "Компонент, таблица или проверка, на которых стоит текущий расчёт."],
  ["open", "Открыта", "s-open", "Перспективная идея: проверена частично, не доведена до расчёта."],
  ["tool", "Инструмент или эталон", "s-tool", "Не входит в расчёт; нужна для проверок, сравнений и сборки."],
  ["superseded", "Заменена", "s-sup", "Работала, но её заменило лучшее решение. Код сохранён."],
  ["rejected", "Отвергнута", "s-rej", "Проверена и оказалась хуже текущего решения."],
  ["not-reproducible", "Не воспроизводится", "s-nr", "Исходные данные или версия кода утрачены."]];
const KIND = {"route": "маршрут", "table-builder": "сборщик таблиц", "validation": "проверка", "probe": "проба",
  "reference": "эталон", "alternative-option": "альтернатива", "tool": "инструмент"};
const ORDER = {order_0: "Прямой свет", order_1: "1 рассеяние", order_ge2: "≥2 рассеяний", order_total: "Всего"};

const ideas = DATA.ideas;
const byId = Object.fromEntries(ideas.map(i => [i.id, i]));
const group = s => s.startsWith("superseded-by:") ? "superseded" : s;
const successor = s => s.startsWith("superseded-by:") ? s.slice(14) : null;
const S = Object.fromEntries(STATUS.map(([k, label, tok, help]) => [k, {label, tok, help}]));
const color = s => `var(--${S[group(s)].tok})`;
const hidden = new Set();
let tab = "tree", query = "", sortKey = "area", sortDir = 1;
try { tab = localStorage.getItem("ideamap.tab") || "tree" } catch (e) {}

const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const pct = x => x == null ? "—" : (Math.abs(x) < 1e-5 && x !== 0 ? (x * 100).toExponential(1) : (x * 100).toFixed(Math.abs(x) < 0.01 ? 3 : 2)) + "%";
const secs = s => s == null ? "" : s < 1 ? s.toFixed(2) + " с" : s < 100 ? s.toFixed(1) + " с" : s < 5400 ? Math.round(s) + " с" : (s / 3600).toFixed(1) + " ч";
const failed = i => i.cases.filter(c => c.status !== "ok").length;

function matches(i) {
  if (hidden.has(group(i.status))) return false;
  if (!query) return true;
  const hay = [i.id, i.name, i.title, i.plain, i.verdict, ...i.modules, ...i.scripts, ...i.tests].join(" ").toLowerCase();
  return query.split(/\s+/).every(w => hay.includes(w));
}

function facts() {
  const m = DATA.meta;
  document.getElementById("facts").innerHTML =
    `<span><b>${m.ideas}</b> идей</span><span><b>${m.cases}</b> замеров</span>` +
    `<span><b>${m.failed}</b> замеров с ошибкой</span><span>замеры ${esc(m.first)} — ${esc(m.last)}</span>`;
}

function legend() {
  const counts = {};
  ideas.forEach(i => counts[group(i.status)] = (counts[group(i.status)] || 0) + 1);
  const total = ideas.length;
  document.getElementById("bar").innerHTML = STATUS.filter(([k]) => counts[k]).map(([k, l, tok]) =>
    `<span style="width:${100 * counts[k] / total}%;background:var(--${tok})" title="${esc(l)}: ${counts[k]}"></span>`).join("");
  document.getElementById("bar").setAttribute("aria-label", STATUS.filter(([k]) => counts[k]).map(([k, l]) => `${l}: ${counts[k]}`).join(", "));
  document.getElementById("filters").innerHTML = STATUS.filter(([k]) => counts[k]).map(([k, l, tok, help]) =>
    `<button class="fchip" data-k="${k}" aria-pressed="${!hidden.has(k)}" title="${esc(help)}" style="--c:var(--${tok})"><span class="dot"></span>${esc(l)} <span class="n">${counts[k]}</span></button>`).join("");
  document.querySelectorAll(".fchip").forEach(b => b.onclick = () => {
    const k = b.dataset.k; hidden.has(k) ? hidden.delete(k) : hidden.add(k); legend(); render();
  });
}

function leaf(i) {
  const main = ["default", "used-by-default"].includes(group(i.status));
  const f = failed(i);
  return `<button class="leaf${main ? " main" : ""}" data-id="${i.id}" style="--c:${color(i.status)}" title="${esc(i.plain || i.title)}">` +
    `<span class="dot"></span><span class="t">${esc(i.name)}</span>${f ? `<span class="x" title="замеры с ошибкой">✕${f}</span>` : ""}</button>`;
}

function tree() {
  const out = AREAS.map(([a, label, about]) => {
    const all = ideas.filter(i => i.area === a), shown = all.filter(matches);
    if (!shown.length) return "";
    const branches = STATUS.map(([k, l, tok]) => {
      const xs = shown.filter(i => group(i.status) === k).sort((x, y) => x.name.localeCompare(y.name, "ru"));
      return xs.length ? `<div class="branch"><div class="blabel"><span class="dot" style="--c:var(--${tok})"></span>${esc(l)} <span class="n">${xs.length}</span></div><div class="leaves">${xs.map(leaf).join("")}</div></div>` : "";
    }).join("");
    return `<section class="area"><div class="anode"><h2>${esc(label)}</h2><span class="cnt">${shown.length} из ${all.length}</span><p>${esc(about)}</p></div><div class="branches">${branches}</div></section>`;
  }).join("");
  return out || `<p class="empty">Ничего не найдено. Измените поиск или включите статусы.</p>`;
}

function evo() {
  const targets = {};
  ideas.filter(i => successor(i.status) && matches(i)).forEach(i => (targets[successor(i.status)] ||= []).push(i));
  const keys = Object.keys(targets).sort((a, b) => targets[b].length - targets[a].length);
  if (!keys.length) return `<p class="empty">Ничего не найдено. Измените поиск или включите статус «Заменена».</p>`;
  return `<p class="hint">Слева идеи, которые работали, но были заменены; справа то, что их заменило и работает сейчас.</p><div class="evo">` + keys.map(t => {
    const n = byId[t];
    return `<div class="chain"><div class="olds">${targets[t].map(i => `<div class="old"><button class="lnk" data-id="${i.id}">${esc(i.name)}</button><span class="p">${esc(i.plain)}</span></div>`).join("")}</div>` +
      `<div class="arrow" aria-hidden="true">→</div><div class="new"><small>заменено на</small><button class="lnk" data-id="${t}">${esc(n ? n.name : t)}</button>${n ? `<span class="st"><span class="dot" style="--c:${color(n.status)}"></span><small>${esc(S[group(n.status)].label)}</small></span>` : ""}</div></div>`;
  }).join("") + `</div>`;
}

function table() {
  const areaName = Object.fromEntries(AREAS.map(([a, l]) => [a, l]));
  const rank = Object.fromEntries(STATUS.map(([k], n) => [k, n]));
  const key = {name: i => i.name, area: i => AREAS.findIndex(a => a[0] === i.area) * 100 + rank[group(i.status)],
    status: i => rank[group(i.status)], cases: i => i.cases.length};
  const rows = ideas.filter(matches).sort((x, y) => {
    const a = key[sortKey](x), b = key[sortKey](y);
    return (typeof a === "string" ? a.localeCompare(b, "ru") : a - b) * sortDir || x.name.localeCompare(y.name, "ru");
  });
  if (!rows.length) return `<p class="empty">Ничего не найдено.</p>`;
  const th = (k, l) => `<th><button data-sort="${k}">${l}${sortKey === k ? (sortDir > 0 ? " ↑" : " ↓") : ""}</button></th>`;
  return `<div class="tablebox"><table><thead><tr>${th("name", "Идея")}${th("area", "Область")}${th("status", "Статус")}${th("cases", "Замеров")}<th>Итог</th></tr></thead><tbody>` +
    rows.map(i => `<tr><td><button class="lnk" data-id="${i.id}">${esc(i.name)}</button></td><td>${esc(areaName[i.area])}</td>` +
      `<td><span class="st"><span class="dot" style="--c:${color(i.status)}"></span>${esc(S[group(i.status)].label)}</span></td>` +
      `<td class="num">${i.cases.length}${failed(i) ? ` <span class="x" style="color:var(--bad)">✕${failed(i)}</span>` : ""}</td><td>${esc(i.plain)}</td></tr>`).join("") +
    `</tbody></table></div>`;
}

function render() {
  const v = document.getElementById("view");
  v.innerHTML = tab === "tree" ? tree() : tab === "evo" ? evo() : table();
  ["tree", "evo", "table"].forEach(t => document.getElementById("tab-" + t).setAttribute("aria-selected", t === tab));
  v.querySelectorAll("[data-id]").forEach(b => b.onclick = () => openIdea(b.dataset.id));
  v.querySelectorAll("[data-sort]").forEach(b => b.onclick = () => {
    const k = b.dataset.sort; sortDir = sortKey === k ? -sortDir : 1; sortKey = k; render();
  });
}

function caseHtml(c) {
  const head = `<div class="chead"><code>${esc(c.name)}</code>` +
    (c.wall_s != null ? `<span class="m">${secs(c.wall_s)}${c.threads === "1" ? ", 1 поток" : c.threads ? ", все потоки" : ""}</span>` : "") +
    (c.wall_vs_baseline != null ? `<span class="m">×${c.wall_vs_baseline.toFixed(2)} к базовому</span>` : "") +
    (c.reference ? `<span class="m">эталон: ${esc(c.reference)}</span>` : "") + `</div>`;
  if (c.status !== "ok") return `<div class="case err">${head}<div class="errtext">Ошибка: ${esc(c.error || c.status)}</div></div>`;
  const ords = Object.keys(ORDER).filter(k => c[k]).map(k =>
    `<div class="ord"><b>${ORDER[k]}</b>смещение ${pct(c[k].charge_bias)}<br>форма L1 ${pct(c[k].bin_l1)}</div>`).join("");
  const mets = c.metrics ? `<details><summary>метрики скрипта</summary><div class="kv">${Object.entries(c.metrics).map(([k, v]) =>
    `<span>${esc(k)}</span><span>${esc(typeof v === "number" ? +v.toPrecision(4) : JSON.stringify(v))}</span>`).join("")}</div></details>` : "";
  return `<div class="case">${head}${ords ? `<div class="orders">${ords}</div>` : ""}${mets}</div>`;
}

function openIdea(id) {
  const i = byId[id]; if (!i) return;
  const d = document.getElementById("drawer");
  const succ = successor(i.status), preds = ideas.filter(x => successor(x.status) === id);
  const list = (title, xs) => xs.length ? `<h4>${title}</h4><ul class="code">${xs.map(x => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  d.innerHTML = `<div class="dhead"><div><h3 id="d-title">${esc(i.name)}</h3><p class="sub">${esc(i.title)}</p></div><button class="close" aria-label="Закрыть">×</button></div>` +
    `<div class="tags"><span class="tag"><span class="dot" style="--c:${color(i.status)}"></span>${esc(S[group(i.status)].label)}</span>` +
    `<span class="tag">${esc(KIND[i.kind] || i.kind)}</span><span class="tag">${esc((AREAS.find(a => a[0] === i.area) || [, i.area])[1])}</span><span class="tag"><code>${esc(i.id)}</code></span></div>` +
    (i.plain ? `<p class="plain">${esc(i.plain)}</p>` : "") +
    (succ ? `<h4>Заменена на</h4><p><button class="lnk" data-id="${succ}">${esc(byId[succ] ? byId[succ].name : succ)}</button></p>` : "") +
    (preds.length ? `<h4>Заменила</h4><p>${preds.map(p => `<button class="lnk" data-id="${p.id}">${esc(p.name)}</button>`).join(", ")}</p>` : "") +
    (i.question ? `<h4>Вопрос</h4><p class="en">${esc(i.question)}</p>` : "") +
    `<h4>Вердикт по замерам</h4><p class="en">${esc(i.verdict)}</p>` +
    (i.reason ? `<h4>Почему не воспроизводится</h4><p class="en">${esc(i.reason)}</p>` : "") +
    (i.parts.length ? `<h4>Что не воспроизводится</h4><ul class="code">${i.parts.map(p => `<li>${esc(typeof p === "string" ? p : Object.values(p).join(": "))}</li>`).join("")}</ul>` : "") +
    (i.cases.length ? `<h4>Замеры (${i.cases.length})</h4><p class="hint">Смещение: ошибка полного заряда; форма L1: суммарная ошибка по временным бинам. Обе относительно эталона.</p><div class="cases">${i.cases.map(caseHtml).join("")}</div>` : "") +
    (i.baseline ? `<h4>Базовый вариант для времени</h4><p><button class="lnk" data-id="${i.baseline}">${esc(byId[i.baseline] ? byId[i.baseline].name : i.baseline)}</button></p>` : "") +
    (i.run ? `<h4>Как запустить</h4><div class="runrow"><pre id="run">${esc(i.run)}</pre><button class="copy" id="copy">Копировать</button></div>` : "") +
    list("Модули", i.modules) + list("Настройки", i.options) + list("Скрипты", i.scripts) + list("Тесты", i.tests);
  d.hidden = false; document.getElementById("scrim").hidden = false;
  d.querySelector(".close").onclick = closeIdea;
  d.querySelectorAll("[data-id]").forEach(b => b.onclick = () => openIdea(b.dataset.id));
  const cp = d.querySelector("#copy");
  if (cp) cp.onclick = () => {
    const t = i.run;
    (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => cp.textContent = "Скопировано", () => {
      const r = document.createRange(); r.selectNodeContents(d.querySelector("#run"));
      const s = getSelection(); s.removeAllRanges(); s.addRange(r); cp.textContent = "Выделено";
    });
  };
  d.scrollTop = 0; d.querySelector(".close").focus();
  try { history.replaceState(null, "", "#" + id) } catch (e) {}
}

function closeIdea() {
  document.getElementById("drawer").hidden = true; document.getElementById("scrim").hidden = true;
  try { history.replaceState(null, "", location.pathname + location.search) } catch (e) {}
}

document.getElementById("scrim").onclick = closeIdea;
document.addEventListener("keydown", e => { if (e.key === "Escape") closeIdea() });
["tree", "evo", "table"].forEach(t => document.getElementById("tab-" + t).onclick = () => {
  tab = t; try { localStorage.setItem("ideamap.tab", t) } catch (e) {} render();
});
document.getElementById("q").oninput = e => { query = e.target.value.trim().toLowerCase(); render() };
facts(); legend(); render();
const h = decodeURIComponent(location.hash.slice(1)); if (byId[h]) openIdea(h);
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "map.html")
    parser.add_argument("--fragment", action="store_true", help="omit the document skeleton")
    args = parser.parse_args()
    args.out.write_text(render(args.fragment))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

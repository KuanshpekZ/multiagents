"""Рисует схему архитектуры docs/architecture.svg (запускать после изменения состава агентов)."""
from pathlib import Path

INK, QUIET, EDGE, DONE, DONE_FILL, ZONE = "#1f2328", "#59636e", "#8c959f", "#0969da", "#ddf4ff", "#f6f8fa"
out = []


def text(x, y, s, size=13, weight=400, fill=INK, anchor="middle"):
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
               f'text-anchor="{anchor}">{s}</text>')


def box(x, y, w, h, title, lines, done):
    stroke, fill, dash = (DONE, DONE_FILL, "") if done else (EDGE, "#ffffff", ' stroke-dasharray="6 4"')
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.5"{dash}/>')
    text(x + w / 2, y + 23, title, 14, 600)
    for i, line in enumerate(lines):
        text(x + w / 2, y + 42 + i * 16, line, 11.5, 400, QUIET)


def arrow(d, dashed=False, label=None, lx=0, ly=0, anchor="start"):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="{EDGE}" stroke-width="1.4"{dash} marker-end="url(#a)"/>')
    if label:
        text(lx, ly, label, 11.5, 400, QUIET, anchor)


W, H = 1000, 800
out.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'font-family="Segoe UI, Arial, sans-serif">')
out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
out.append(f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{EDGE}"/></marker></defs>')
text(24, 32, "Шлюз поднимает алерты, пять агентов разбирают их на сервере", 17, 600, INK, "start")

# --- зона шлюза ---
out.append(f'<rect x="24" y="50" width="952" height="104" rx="10" fill="{ZONE}" stroke="{EDGE}"/>')
text(40, 72, "Шлюз IoT-сети: лёгкий уровень, без LLM", 13, 600, QUIET, "start")
box(56, 82, 230, 58, "IoT-устройства", ["датчики и другие устройства"], True)
box(380, 82, 300, 58, "LightGBM IDS", ["11 поведенческих признаков, без IP и портов"], True)
arrow("M286 111H380", label="весь трафик", lx=333, ly=103, anchor="middle")

# --- зона сервера ---
out.append(f'<rect x="24" y="196" width="952" height="580" rx="10" fill="#ffffff" stroke="{EDGE}"/>')
text(40, 219, "Сервер мониторинга: конвейер из пяти агентов", 13, 600, QUIET, "start")
arrow("M530 140V236", label="потоки с вердиктом и алерты", lx=542, ly=180)

box(250, 236, 360, 74, "1. Триаж", ["fetch_gateway_alerts · group_alerts",
                                    "get_flow_features · traffic_baseline"], True)
box(56, 352, 300, 74, "2. Объяснение (XAI)", ["shap_incident · shap_flow", "feature_reference"], True)
box(404, 352, 300, 74, "3. Проверка устойчивости", ["run_hopskipjump · check_constraints",
                                                    "perturbation_report"], False)
box(250, 468, 360, 74, "4. Реагирование", ["attack_lookup · cve_search · playbook_lookup",
                                           "generate_firewall_rule · validate_rule"], False)
box(250, 584, 360, 74, "5. Критик", ["verify_mitre_id · verify_shap_claims",
                                     "consistency_rules"], False)
box(690, 592, 150, 58, "Отчёт", ["по инцидентам"], True)

arrow("M430 310V332H206V352", label="ExplainTask", lx=214, ly=327)
arrow("M430 332H554V352", dashed=True, label="RedTeamTask", lx=562, ly=327)
out.append(f'<path d="M206 426V446H554V426" fill="none" stroke="{EDGE}" stroke-width="1.4"/>')
arrow("M430 446V468", label="Explanation + RobustnessReport", lx=440, ly=461)
arrow("M430 542V584", dashed=True, label="ResponsePlan", lx=440, ly=568)
arrow("M610 621H690", dashed=True)
text(650, 612, "Verdict", 11.5, 400, QUIET)
# петля доработки
out.append(f'<path d="M250 621H40V389H56" fill="none" stroke="{EDGE}" stroke-width="1.4" '
           f'stroke-dasharray="5 4" marker-end="url(#a)"/>')
out.append(f'<path d="M40 505H250" fill="none" stroke="{EDGE}" stroke-width="1.4" stroke-dasharray="5 4" '
           f'marker-end="url(#a)"/>')
text(145, 612, "на доработку, не более 2 раз", 11.5, 400, QUIET)

# --- внешние источники ---
out.append(f'<rect x="740" y="236" width="220" height="306" rx="8" fill="{ZONE}" stroke="{EDGE}"/>')
text(850, 260, "Данные и внешние источники", 13, 600)
for i, (name, used) in enumerate([("Датасет TON_IoT (файлы)", "сценарии трафика"),
                                  ("LightGBM, SHAP, ART", "выполнение кода: агенты 1–3"),
                                  ("MITRE ATT&amp;CK", "локальная база: агенты 4, 5"),
                                  ("NVD API", "внешний API: агент 4"),
                                  ("LLM API", "все агенты, ключ в .env")]):
    y = 292 + i * 50
    text(756, y, name, 12.5, 600, INK, "start")
    text(756, y + 17, used, 11.5, 400, QUIET, "start")

# --- состояние и журнал ---
out.append(f'<rect x="56" y="686" width="904" height="40" rx="8" fill="{ZONE}" stroke="{EDGE}"/>')
text(508, 711, "Конвейер (код, не агент): порядок шагов, лимиты, общее состояние state.json, журнал calls.jsonl", 13)

# --- легенда ---
out.append(f'<rect x="56" y="742" width="22" height="16" rx="4" fill="{DONE_FILL}" stroke="{DONE}" stroke-width="1.5"/>')
text(86, 755, "работает в прототипе (Ассайнмент 2)", 12, 400, QUIET, "start")
out.append(f'<rect x="340" y="742" width="22" height="16" rx="4" fill="#ffffff" stroke="{EDGE}" stroke-width="1.5" '
           f'stroke-dasharray="6 4"/>')
text(370, 755, "спроектировано, реализация в Ассайнменте 4", 12, 400, QUIET, "start")
out.append("</svg>")

target = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"
target.write_text("\n".join(out), encoding="utf-8")
print("Сохранено:", target)

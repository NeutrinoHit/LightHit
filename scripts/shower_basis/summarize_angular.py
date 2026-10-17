"""Summarise study_angular.py output: worst over test events per variant."""
import json
import sys
from collections import defaultdict

rows = defaultdict(list)
for line in open(sys.argv[1]):
    if not line.startswith("{"):
        continue
    item = json.loads(line)
    if "variant" not in item:
        continue
    key = (item["variant"], str(item.get("rank", item.get("bins", item.get("M", "")))))
    rows[key].append(item)
print(f"{'variant':16s} {'param':8s} {'src L2':>7s} {'q>=2 bright':>11s} {'q>=2 all':>9s} "
      f"{'abs pe':>8s} {'L1 max':>7s} {'L1 med':>7s}")
for (variant, param), items in rows.items():
    worst = lambda name: max(i[name] for i in items)  # noqa: E731
    print(f"{variant:16s} {param:8s} {worst('source_l2'):7.3f} {worst('ge2_rel_max_bright'):11.4f} "
          f"{worst('ge2_rel_max'):9.4f} {worst('ge2_abs_max'):8.4f} {worst('ge2_L1_max'):7.3f} "
          f"{max(i['ge2_L1_median'] for i in items):7.4f}")

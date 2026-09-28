"""Markdown rows of the validation table from a benchmark summary.json."""
import json
import sys

summary = json.load(open(sys.argv[1]))
for spec, r in summary["comparisons"].items():
    print(f"| {spec.split(':')[1]} | {r['active_reference']} | {r['active_lost']}/{r['active_gained']} | "
          f"{r['total_charge_rel_max_bright']:.1e} | {r['order_ge2_charge_rel_max']:.1e} | "
          f"{r['order_ge2_charge_abs_max_pe']:.1e} ф.э. | {100 * r['total_L1_max_bright']:.2f}% / "
          f"{100 * r['total_L1_max']:.2f}% | {100 * r['order_ge2_L1_max']:.2f}% | "
          f"{r['early_light_rel_max_bright']:.1e} |")

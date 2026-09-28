"""Print the accuracy table of a shower_basis_benchmark summary.json."""
import json
import sys

summary = json.load(open(sys.argv[1]))
keys = [("active_reference", "act.ref", "d"), ("active_lost", "lost", "d"), ("active_gained", "gain", "d"),
        ("total_charge_rel_max_bright", "tot q br", ".1e"), ("order_ge2_charge_rel_max_bright", "q>=2 br", ".1e"),
        ("order_ge2_charge_rel_max", "q>=2 all", ".1e"), ("order_ge2_charge_abs_max_pe", "q>=2 abs", ".1e"),
        ("total_L1_max_bright", "L1 tot br", ".1e"), ("total_L1_max", "L1 tot", ".1e"),
        ("order_ge2_L1_max_bright", "L1>=2 br", ".1e"), ("order_ge2_L1_max", "L1>=2", ".1e"),
        ("early_light_abs_max_pe", "early abs", ".1e"), ("order0_charge_rel_max", "q0", ".0e"),
        ("order1_charge_rel_max", "q1", ".0e")]
print("event    " + " ".join(f"{label:>9s}" for _, label, _ in keys))
for event, row in summary["comparisons"].items():
    print(f"{event:8s} " + " ".join(f"{row[k]:>9{fmt}}" for k, _, fmt in keys))
for name, run in summary["runs"].items():
    for entry in run["events"]:
        t = entry["timings"]
        print(name.split(":")[0], entry["kind"], entry["event"], f"{entry['seconds']:.1f}s",
              " ".join(f"{k.replace('_seconds', '')}={v:.1f}" for k, v in t.items()
                       if k not in ("elapsed_seconds", "rte_elapsed_seconds")),
              f"peak={entry['peak_rss_mib']:.0f}MiB points={(entry.get('basis') or {}).get('cells')}")

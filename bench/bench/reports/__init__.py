"""Cross-walk + render the result store (ADR-0017 §"Reporting").

``compare`` answers "head vs base on this machine"; ``longitudinal``
answers "how has perf moved over the last N days". Both consume the
result tree at ``results/<git-sha>/<machine-fp>/<workload>/<iou>/<impl>.json``
via :mod:`bench.reports.load`.
"""

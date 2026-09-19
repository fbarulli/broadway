from __future__ import annotations

from broadway.lineage.models import LineageGraph


def to_mermaid(graph: LineageGraph) -> str:
    lines = ["flowchart LR"]
    for node in graph.nodes:
        safe = node.id.replace(":", "_").replace("-", "_")
        label = node.label
        if node.sample_name is not None:
            label = f"{node.label} (sample={node.sample_name}, role={node.sample_role})"
        lines.append(f'    {safe}["{label}"]')
    for edge in graph.edges:
        s = edge.source.replace(":", "_").replace("-", "_")
        t = edge.target.replace(":", "_").replace("-", "_")
        lines.append(f"    {s} -->|{edge.relation}| {t}")
    return "\n".join(lines)

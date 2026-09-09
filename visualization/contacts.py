#!/usr/bin/env python3
"""Generate cumulative, temporal, and per-minute contact visualizations."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "contacts" / "contacts.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "contacts"
INTERVAL_SECONDS = 60
MINUTE_SECONDS = 60
FIVE_MINUTE_BIN_SECONDS = 5 * MINUTE_SECONDS
CLUSTERING_BIN_SECONDS = 30 * MINUTE_SECONDS
DEFAULT_GIF_FPS = 120.0
DEFAULT_GIF_HOLD_SECONDS = 30


def load_contacts(path: Path) -> pd.DataFrame:
    contacts = pd.read_parquet(path, columns=["t", "i", "j"])
    if contacts.empty:
        raise ValueError("contact dataset is empty")
    if contacts[["t", "i", "j"]].isnull().any().any():
        raise ValueError("contact dataset contains null values")
    if (contacts["i"] == contacts["j"]).any():
        raise ValueError("contact dataset contains self-contacts")
    return contacts


def cumulative_graph(contacts: pd.DataFrame) -> nx.Graph:
    pairs = pd.DataFrame(
        {
            "source": contacts[["i", "j"]].min(axis=1),
            "target": contacts[["i", "j"]].max(axis=1),
        }
    )
    weights = (
        pairs.value_counts(sort=False)
        .rename("weight")
        .reset_index()
    )

    graph = nx.Graph()
    graph.add_weighted_edges_from(
        weights[["source", "target", "weight"]].itertuples(
            index=False, name=None
        )
    )
    return graph


def plot_cumulative_network(
    contacts: pd.DataFrame,
    graph: nx.Graph,
    positions: dict[int, np.ndarray],
    output: Path,
) -> None:
    nodes = list(graph.nodes)
    strengths = np.asarray(
        [graph.degree(node, weight="weight") for node in nodes],
        dtype=float,
    )
    edge_weights = np.asarray(
        [data["weight"] for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    node_sizes = 12.0 + 90.0 * np.sqrt(strengths / strengths.max())
    edge_widths = 0.1 + 1.8 * (
        np.log1p(edge_weights) / np.log1p(edge_weights.max())
    )

    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=edge_widths,
        edge_color="#34495e",
        alpha=0.10,
    )
    node_collection = nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        ax=ax,
        node_size=node_sizes,
        node_color=strengths,
        cmap="viridis",
        linewidths=0.2,
        edgecolors="white",
    )
    node_collection.set_norm(
        LogNorm(vmin=max(1.0, strengths.min()), vmax=strengths.max())
    )
    colorbar = fig.colorbar(node_collection, ax=ax, shrink=0.68, pad=0.01)
    colorbar.set_label("Cumulative active contact intervals per person")

    ax.set_title(
        "Cumulative contact network\n"
        f"{graph.number_of_nodes():,} people, "
        f"{graph.number_of_edges():,} distinct pairs, "
        f"{len(contacts):,} active {INTERVAL_SECONDS}-second intervals",
        fontsize=15,
        pad=14,
    )
    ax.text(
        0.5,
        -0.01,
        "Node size and color represent weighted degree; "
        "edge width represents cumulative pair contacts.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.set_axis_off()
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def temporal_graphs(
    contacts: pd.DataFrame,
    nodes: list[int],
    bin_seconds: int,
    *,
    cumulative: bool = False,
) -> list[tuple[int, int, nx.Graph]]:
    bin_end = (
        (contacts["t"] - 1) // bin_seconds + 1
    ) * bin_seconds
    groups = {
        int(end): group
        for end, group in contacts.groupby(bin_end, sort=True)
    }
    first_end = int(bin_end.min())
    last_end = int(bin_end.max())

    frames = []
    accumulated = nx.Graph()
    accumulated.add_nodes_from(nodes)
    for end in range(
        first_end,
        last_end + bin_seconds,
        bin_seconds,
    ):
        frame_contacts = groups.get(end)
        if frame_contacts is None:
            increment = nx.Graph()
        else:
            increment = cumulative_graph(frame_contacts)
        if cumulative:
            for source, target, data in increment.edges(data=True):
                if accumulated.has_edge(source, target):
                    accumulated[source][target]["weight"] += data["weight"]
                else:
                    accumulated.add_edge(source, target, weight=data["weight"])
            graph = accumulated.copy()
        else:
            graph = increment
            graph.add_nodes_from(nodes)
        frames.append((end - bin_seconds, end, graph))
    return frames


def elapsed_label(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3_600)
    minutes = remainder // MINUTE_SECONDS
    return f"{hours:03d}h{minutes:02d}"


def plot_temporal_networks(
    contacts: pd.DataFrame,
    cumulative: nx.Graph,
    positions: dict[int, np.ndarray],
    output_dir: Path,
    *,
    bin_seconds: int,
    cumulative_growth: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = list(cumulative.nodes)
    frames = temporal_graphs(
        contacts,
        nodes,
        bin_seconds,
        cumulative=cumulative_growth,
    )
    for existing_frame in output_dir.glob("contacts_*.png"):
        existing_frame.unlink()

    cumulative_centrality = nx.degree_centrality(cumulative)
    centrality_values = np.asarray(
        [cumulative_centrality[node] for node in nodes],
        dtype=float,
    )
    base_sizes = {
        node: size
        for node, size in zip(
            nodes,
            8.0 + 42.0 * np.sqrt(
                centrality_values / centrality_values.max()
            ),
        )
    }
    max_node_strength = max(
        graph.degree(node, weight="weight")
        for _, _, graph in frames
        for node in nodes
    )
    max_edge_weight = max(
        (
            data["weight"]
            for _, _, graph in frames
            for _, _, data in graph.edges(data=True)
        ),
        default=1,
    )
    color_norm = LogNorm(vmin=1, vmax=max_node_strength)
    color_map = plt.get_cmap("viridis")
    x_coordinates = np.asarray(
        [positions[node][0] for node in nodes],
        dtype=float,
    )
    y_coordinates = np.asarray(
        [positions[node][1] for node in nodes],
        dtype=float,
    )
    x_margin = max(0.05, np.ptp(x_coordinates) * 0.06)
    y_margin = max(0.05, np.ptp(y_coordinates) * 0.06)
    x_limits = (
        float(x_coordinates.min() - x_margin),
        float(x_coordinates.max() + x_margin),
    )
    y_limits = (
        float(y_coordinates.min() - y_margin),
        float(y_coordinates.max() + y_margin),
    )

    for frame_number, (start, end, graph) in enumerate(frames):
        active_nodes = [
            node for node in nodes if graph.degree(node, weight="weight") > 0
        ]
        edge_weights = np.asarray(
            [data["weight"] for _, _, data in graph.edges(data=True)],
            dtype=float,
        )

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_axes((0.04, 0.04, 0.80, 0.86))
        colorbar_ax = fig.add_axes((0.88, 0.18, 0.025, 0.60))
        background_sizes = (
            [8.0] * len(nodes)
            if cumulative_growth
            else [base_sizes[node] for node in nodes]
        )
        nx.draw_networkx_nodes(
            cumulative,
            positions,
            nodelist=nodes,
            ax=ax,
            node_size=background_sizes,
            node_color="#d4dce2",
            linewidths=0,
            alpha=0.45,
        )
        if graph.number_of_edges():
            edge_widths = 0.2 + 2.8 * (
                np.log1p(edge_weights) / np.log1p(max_edge_weight)
            )
            nx.draw_networkx_edges(
                graph,
                positions,
                ax=ax,
                width=edge_widths,
                edge_color="#34495e",
                alpha=0.22,
            )
        if active_nodes:
            if cumulative_growth:
                active_sizes = [
                    8.0
                    + 42.0
                    * np.sqrt(
                        graph.degree(node, weight="weight") / max_node_strength
                    )
                    for node in active_nodes
                ]
            else:
                active_sizes = [base_sizes[node] for node in active_nodes]
            nx.draw_networkx_nodes(
                graph,
                positions,
                nodelist=active_nodes,
                ax=ax,
                node_size=active_sizes,
                node_color=[
                    color_map(
                        color_norm(graph.degree(node, weight="weight"))
                    )
                    for node in active_nodes
                ],
                linewidths=0.2,
                edgecolors="white",
            )

        colorbar = fig.colorbar(
            ScalarMappable(norm=color_norm, cmap=color_map),
            cax=colorbar_ax,
        )
        if cumulative_growth:
            colorbar.set_label(
                "Cumulative active "
                f"{INTERVAL_SECONDS}-second contact intervals per person"
            )
            first_start = frames[0][0]
            ax.set_title(
                "Cumulative contact network: "
                f"{elapsed_label(first_start)}–{elapsed_label(end)}\n"
                f"{len(active_nodes):,} people, "
                f"{graph.number_of_edges():,} distinct pairs, "
                f"{int(graph.size(weight='weight')):,} contact intervals",
                fontsize=13,
                pad=12,
            )
        else:
            colorbar.set_label(
                f"Active {INTERVAL_SECONDS}-second contact intervals per person"
            )
            ax.set_title(
                f"Contact network: {elapsed_label(start)}–{elapsed_label(end)}\n"
                f"{len(active_nodes):,} active people, "
                f"{graph.number_of_edges():,} active pairs, "
                f"{int(graph.size(weight='weight')):,} contact intervals",
                fontsize=13,
                pad=12,
            )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        if cumulative_growth:
            caption = (
                "Node size and color: cumulative activity so far; "
                "edges accumulate over time"
            )
        else:
            caption = (
                "Node size: cumulative degree centrality; "
                "node color: activity in this bin"
            )
        fig.text(
            0.44,
            0.015,
            caption,
            ha="center",
            va="bottom",
            fontsize=8,
        )

        output = output_dir / (
            f"contacts_{frame_number:04d}_"
            f"{elapsed_label(start)}_to_{elapsed_label(end)}.png"
        )
        fig.savefig(output, dpi=160, facecolor="white")
        plt.close(fig)

    return len(frames)


def contacts_per_minute(contacts: pd.DataFrame) -> pd.Series:
    minute_end = (
        (contacts["t"] - 1) // MINUTE_SECONDS + 1
    ) * MINUTE_SECONDS
    counts = minute_end.value_counts().sort_index()
    complete_minutes = pd.RangeIndex(
        start=int(counts.index.min()),
        stop=int(counts.index.max()) + MINUTE_SECONDS,
        step=MINUTE_SECONDS,
        name="minute_end",
    )
    return counts.reindex(complete_minutes, fill_value=0).astype("int64")


def plot_contacts_per_minute(contacts: pd.DataFrame, output: Path) -> None:
    counts = contacts_per_minute(contacts)
    hours = counts.index.to_numpy(dtype=float) / 3_600

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.bar(
        hours,
        counts.to_numpy(),
        width=1 / 60,
        color="#2878b5",
        linewidth=0,
    )
    ax.set_title("Active contacts per one-minute bin", fontsize=15, pad=10)
    ax.set_xlabel("Elapsed collection time (hours; bins labeled by end time)")
    ax.set_ylabel("Active contact records per minute")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(hours.min() - 1 / 60, hours.max() + 1 / 60)
    ax.text(
        1.0,
        1.01,
        f"{len(counts):,} one-minute bins; empty bins shown as zero",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_clustering_over_time(
    contacts: pd.DataFrame,
    cumulative: nx.Graph,
    output: Path,
) -> None:
    frames = temporal_graphs(
        contacts,
        list(cumulative.nodes),
        CLUSTERING_BIN_SECONDS,
    )
    midpoint_hours = []
    clustering = []

    for start, end, graph in frames:
        active_nodes = [
            node for node, degree in graph.degree() if degree > 0
        ]
        midpoint_hours.append((start + end) / 2 / 3_600)
        if active_nodes:
            active_graph = graph.subgraph(active_nodes)
            clustering.append(nx.average_clustering(active_graph))
        else:
            clustering.append(np.nan)

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(
        midpoint_hours,
        clustering,
        color="#2878b5",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )
    ax.set_title(
        "Average network clustering by 30-minute bin",
        fontsize=15,
        pad=10,
    )
    ax.set_xlabel("Elapsed collection time (hours; points at bin midpoints)")
    ax.set_ylabel("Average local clustering coefficient")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.text(
        1.0,
        1.01,
        "Computed on active people using unweighted edges; "
        "empty bins shown as gaps",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def plot_degree_centrality_distribution(
    graph: nx.Graph,
    output: Path,
) -> None:
    centrality = np.asarray(
        list(nx.degree_centrality(graph).values()),
        dtype=float,
    )
    mean = float(centrality.mean())
    median = float(np.median(centrality))

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.hist(
        centrality,
        bins="fd",
        color="#2878b5",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(
        mean,
        color="#c44e52",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean:.3f}",
    )
    ax.axvline(
        median,
        color="#4c4c4c",
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {median:.3f}",
    )
    ax.set_title(
        "Degree centrality in the cumulative contact network",
        fontsize=15,
        pad=10,
    )
    ax.set_xlabel(
        f"Degree centrality (distinct contacts divided by {len(graph) - 1})"
    )
    ax.set_ylabel("Number of people")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)


def write_contact_gif(
    frame_dir: Path,
    output: Path,
    *,
    fps: float,
    final_hold_seconds: float,
    glob_pattern: str = "contacts_*.png",
) -> None:
    frames = sorted(frame_dir.glob(glob_pattern))
    if not frames:
        raise ValueError(f"no frames matching {glob_pattern} in {frame_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_complex = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={final_hold_seconds},"
        "scale=800:-1:flags=lanczos,split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frame_dir / glob_pattern),
        "-filter_complex",
        filter_complex,
        "-loop",
        "0",
        str(output),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"contacts Parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PNG output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for the network layout (default: 42)",
    )
    parser.add_argument(
        "--growing-gif",
        action="store_true",
        help="also write a fast cumulative-growth GIF",
    )
    parser.add_argument(
        "--growing-gif-only",
        action="store_true",
        help="write only the cumulative-growth GIF",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=DEFAULT_GIF_FPS,
        help=f"growth-GIF frames per second (default: {DEFAULT_GIF_FPS:g})",
    )
    parser.add_argument(
        "--gif-hold-seconds",
        type=float,
        default=DEFAULT_GIF_HOLD_SECONDS,
        help=(
            "seconds to hold the final cumulative network "
            f"(default: {DEFAULT_GIF_HOLD_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--gif-bin-seconds",
        type=int,
        default=MINUTE_SECONDS,
        help=(
            "bin width in seconds for the growing cumulative GIF "
            f"(default: {MINUTE_SECONDS})"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contacts = load_contacts(args.input)
    graph = cumulative_graph(contacts)
    positions = nx.spring_layout(
        graph,
        seed=args.seed,
        weight="weight",
        iterations=75,
    )

    if not args.growing_gif_only:
        network_output = args.output_dir / "contacts_network.png"
        timeline_output = args.output_dir / "contacts_per_minute.png"
        clustering_output = args.output_dir / "contacts_clustering_over_time.png"
        degree_output = (
            args.output_dir / "contacts_degree_centrality_distribution.png"
        )
        temporal_5min_output_dir = args.output_dir / "temporal_5min"
        temporal_1min_output_dir = args.output_dir / "temporal_1min"
        plot_cumulative_network(contacts, graph, positions, network_output)
        plot_contacts_per_minute(contacts, timeline_output)
        plot_clustering_over_time(contacts, graph, clustering_output)
        plot_degree_centrality_distribution(graph, degree_output)
        temporal_5min_frames = plot_temporal_networks(
            contacts,
            graph,
            positions,
            temporal_5min_output_dir,
            bin_seconds=FIVE_MINUTE_BIN_SECONDS,
        )
        temporal_1min_frames = plot_temporal_networks(
            contacts,
            graph,
            positions,
            temporal_1min_output_dir,
            bin_seconds=MINUTE_SECONDS,
        )

        print(f"Wrote {network_output}")
        print(f"Wrote {timeline_output}")
        print(f"Wrote {clustering_output}")
        print(f"Wrote {degree_output}")
        print(
            f"Wrote {temporal_5min_frames} frames to "
            f"{temporal_5min_output_dir}"
        )
        print(
            f"Wrote {temporal_1min_frames} frames to "
            f"{temporal_1min_output_dir}"
        )

    if args.growing_gif or args.growing_gif_only:
        growing_dir = args.output_dir / (
            f"cumulative_{args.gif_bin_seconds}s"
        )
        growing_frames = plot_temporal_networks(
            contacts,
            graph,
            positions,
            growing_dir,
            bin_seconds=args.gif_bin_seconds,
            cumulative_growth=True,
        )
        gif_output = args.output_dir / "contacts_cumulative_growing.gif"
        write_contact_gif(
            growing_dir,
            gif_output,
            fps=args.gif_fps,
            final_hold_seconds=args.gif_hold_seconds,
        )
        print(f"Wrote {growing_frames} frames to {growing_dir}")
        print(
            f"Wrote {gif_output} at {args.gif_fps:g} fps "
            f"with a {args.gif_hold_seconds:g}s final hold"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

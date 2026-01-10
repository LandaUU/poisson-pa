import networkx as nx
from pathlib import Path


def export_for_gnuplot(G: nx.Graph, out_dir=".", seed=42):
    """
    Создаёт файлы nodes.dat, edges.dat и labels.dat для отрисовки графа G в gnuplot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(G, seed=seed)

    # --- nodes.dat ---
    with (out_dir / "nodes.dat").open("w") as fn:
        for u, (x, y) in pos.items():
            fn.write(f"{x:.6f}\t{y:.6f}\n")

    # --- edges.dat ---
    with (out_dir / "edges.dat").open("w") as fe:
        for u, v in G.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            fe.write(f"{x1:.6f}\t{y1:.6f}\t{x2 - x1:.6f}\t{y2 - y1:.6f}\n")

    # --- labels.dat ---
    with (out_dir / "labels.dat").open("w") as fl:
        for u, (x, y) in pos.items():
            fl.write(f'{x:.6f}\t{y:.6f}\t"{u}"\n')

    print(f"Файлы сохранены в {out_dir.resolve()}")
    print("\nГотовая команда для gnuplot:\n")
    print(f"""gnuplot -e "
set terminal pngcairo size 1920,1080;
set output 'graph.png';
unset key; unset xtics; unset ytics; set border 0; set size ratio -1;
set pointsize 2;
plot \\
  '{out_dir}/edges.dat' using 1:2:3:4 with vectors nohead lw 2 lc rgb '#ff0028', \\
  '{out_dir}/nodes.dat' using 1:2 with points pt 7 lc rgb '#1c1b1b'
"
""")


# пример использования:
if __name__ == "__main__":
    G = nx.fast_gnp_random_graph(20, 0.15, directed=True)
    export_for_gnuplot(G, out_dir="gnuplot_out")

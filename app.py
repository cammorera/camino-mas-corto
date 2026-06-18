import streamlit as st
import heapq
import time
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO
import re

st.set_page_config(
    page_title="Camino más corto — Redes",
    page_icon="🗺️",
    layout="wide",
)

# ── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.02em; color: #0f172a; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1e293b; margin-top: 0 !important; }
h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #334155; }

.tag {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    padding: 2px 8px;
    color: #475569;
    margin: 2px 3px;
}

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.72rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 1.5rem; font-weight: 600; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
.metric-card .sub { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

.path-row {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #166534;
    margin-bottom: 0.4rem;
}
.pair-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0.7rem 0 0.3rem;
}

.matrix-table { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

.section-rule {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0 1rem;
}

.stAlert { border-radius: 6px !important; }
.stButton > button {
    background: #0f172a !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: background 0.2s;
}
.stButton > button:hover { background: #1e3a5f !important; }

.stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_network_file(content: str):
    """Parse Red1.nf: lines with 'origin dest distance flag'"""
    graph = {}
    lines = content.strip().splitlines()
    arc_count = 0
    for line in lines[1:]:          # skip header line
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            u, v, d = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        if u not in graph:
            graph[u] = []
        graph[u].append((v, d))
        arc_count += 1
    return graph, arc_count


def parse_individual_file(content: str):
    """
    Parse Individual_2.txt compact format.
    Returns: i, o, n, arcos_r (list of (u,v,d)), arcos_s (list of (u,v,d))
    """
    tokens = content.split()
    idx = 0
    i_val = o_val = n_val = None
    arcos_r, arcos_s = [], []

    while idx < len(tokens):
        t = tokens[idx]
        if t == 'i':
            i_val = int(tokens[idx + 1]); idx += 2
        elif t == 'o':
            o_val = int(tokens[idx + 1]); idx += 2
        elif t == 'n':
            n_val = int(tokens[idx + 1]); idx += 2
        elif t == 'r':
            u, v, d = int(tokens[idx+1]), int(tokens[idx+2]), float(tokens[idx+3])
            arcos_r.append((u, v, d)); idx += 4
        elif t == 's':
            u, v, d = int(tokens[idx+1]), int(tokens[idx+2]), float(tokens[idx+3])
            arcos_s.append((u, v, d)); idx += 4
        else:
            idx += 1

    return i_val, o_val, n_val, arcos_r, arcos_s


def extract_delivery_nodes(arcos_s):
    """Delivery nodes = unique origins in service arcs. Depot = 0."""
    nodes = sorted(set(u for u, v, d in arcos_s))
    return nodes


# ── Dijkstra ──────────────────────────────────────────────────────────────────
def dijkstra(graph, source):
    dist = {source: 0}
    prev = {source: None}
    pq = [(0, source)]
    visited = set()

    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            new_cost = cost + w
            if v not in dist or new_cost < dist[v]:
                dist[v] = new_cost
                prev[v] = u
                heapq.heappush(pq, (new_cost, v))
    return dist, prev


def reconstruct_path(prev, source, target):
    path = []
    node = target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    if path and path[0] == source:
        return path
    return []


# ── Visualización de la red pequeña ──────────────────────────────────────────
def draw_small_network(arcos_r, arcos_s, delivery_nodes, paths_found=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')

    G = nx.DiGraph()
    for u, v, d in arcos_r:
        G.add_edge(u, v, weight=d, kind='r')
    for u, v, d in arcos_s:
        G.add_edge(u, v, weight=d, kind='s')

    all_nodes = list(G.nodes())
    pos = nx.spring_layout(G, seed=42, k=1.8)

    depot_nodes = [0] if 0 in all_nodes else []
    client_nodes = [n for n in delivery_nodes if n in all_nodes]
    other_nodes = [n for n in all_nodes if n not in client_nodes and n not in depot_nodes]

    r_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('kind') == 'r']
    s_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('kind') == 's']

    nx.draw_networkx_edges(G, pos, edgelist=r_edges, ax=ax,
        edge_color='#94a3b8', arrows=True, arrowsize=10,
        width=1.2, connectionstyle='arc3,rad=0.05', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=s_edges, ax=ax,
        edge_color='#6366f1', arrows=True, arrowsize=12,
        width=1.8, connectionstyle='arc3,rad=0.1', style='dashed', alpha=0.85)

    if other_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, ax=ax,
            node_color='#e2e8f0', node_size=500, linewidths=1.2, edgecolors='#94a3b8')
    if depot_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=depot_nodes, ax=ax,
            node_color='#0f172a', node_size=700, linewidths=0, edgecolors='none')
    if client_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=client_nodes, ax=ax,
            node_color='#22c55e', node_size=650, linewidths=1.5, edgecolors='#166534')

    labels = {n: str(n) for n in all_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
        font_size=8, font_color='white' if depot_nodes else '#1e293b',
        font_weight='bold')
    # Fix label colors per node type
    for node, (x, y) in pos.items():
        color = 'white' if node in depot_nodes else ('#166534' if node in client_nodes else '#334155')
        ax.text(x, y, str(node), ha='center', va='center',
                fontsize=8, fontweight='bold', color=color)

    edge_labels = {(u, v): str(int(d['weight'])) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
        font_size=7, font_color='#64748b', bbox=dict(boxstyle='round,pad=0.1', fc='#f8fafc', alpha=0.6))

    legend_elements = [
        mpatches.Patch(color='#0f172a', label='Depósito (0)'),
        mpatches.Patch(color='#22c55e', label='Nodo de entrega'),
        mpatches.Patch(color='#e2e8f0', label='Nodo intermedio'),
        plt.Line2D([0],[0], color='#94a3b8', lw=1.5, label='Arco red (r)'),
        plt.Line2D([0],[0], color='#6366f1', lw=1.8, ls='dashed', label='Arco servicio (s)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7,
              framealpha=0.9, edgecolor='#e2e8f0')

    ax.set_title('Red de servicio (Individual_2.txt)', fontsize=9,
                 color='#334155', pad=8, fontfamily='sans-serif')
    ax.axis('off')
    plt.tight_layout()
    return fig


# ── Visualización de la matriz ────────────────────────────────────────────────
def display_matrix(matrix, nodes_labels):
    n = len(nodes_labels)
    data = {}
    for j_idx, j in enumerate(nodes_labels):
        col = []
        for i_idx, i in enumerate(nodes_labels):
            val = matrix.get((i, j), None)
            if i == j:
                col.append("0")
            elif val is None:
                col.append("∞")
            else:
                col.append(f"{val:,.1f}")
        data[f"→ {j}"] = col
    df = pd.DataFrame(data, index=[f"{l} ↓" for l in nodes_labels])
    return df


# ── Generar código AMPL ───────────────────────────────────────────────────────
def generate_ampl_mod():
    return """\
# shortest_path.mod
# Modelo de flujo para camino más corto (un par origen-destino a la vez)

set NODOS;
set ARCOS within NODOS cross NODOS;

param dist {ARCOS} >= 0;
param origen symbolic in NODOS;
param destino symbolic in NODOS;

var f {ARCOS} >= 0;

minimize total_dist:
    sum {(i,j) in ARCOS} dist[i,j] * f[i,j];

s.t. balance {k in NODOS}:
    sum {(k,j) in ARCOS} f[k,j]
  - sum {(i,k) in ARCOS} f[i,k]
  = if k = origen then  1
    else if k = destino then -1
    else 0;
"""


def generate_ampl_dat(graph, delivery_nodes, depot=0):
    all_points = [depot] + delivery_nodes
    nodes_str = " ".join(str(n) for n in sorted(set(
        [n for arc in graph for n in [arc] + [v for v, _ in graph[arc]]]
    ))[:200])  # muestra primeros 200 nodos para no saturar

    arcs_lines = []
    for u in list(graph.keys())[:500]:
        for v, d in graph[u]:
            arcs_lines.append(f"  {u} {v}  {d}")

    return f"""\
# shortest_path.dat
# Datos generados automáticamente

set NODOS := {nodes_str} ... ;  # Todos los nodos de Red1.nf

set ARCOS :=
  # (fragmento — usar Red1.nf completo)
{chr(10).join(arcs_lines[:80])}
  ... ;

param dist :=
{chr(10).join(f"  {u} {v}  {d}" for line in arcs_lines[:80] for u,v,d in [line.strip().split()])}
  ... ;

# Cambiar origen/destino para cada par:
param origen := {all_points[0]} ;
param destino := {all_points[1]} ;
"""


def generate_run_script(delivery_nodes, depot=0):
    all_points = [depot] + delivery_nodes
    pairs = [(all_points[i], all_points[j])
             for i in range(len(all_points))
             for j in range(len(all_points)) if i != j]
    lines = ["# shortest_path.run", "model shortest_path.mod;", "data shortest_path.dat;", ""]
    for orig, dest in pairs:
        lines += [
            f"let origen := {orig};",
            f"let destino := {dest};",
            f"solve;",
            f'printf "D[%d][%d] = %g\\n", {orig}, {dest}, total_dist;',
            "",
        ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# UI principal
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🗺️ Camino más corto — Optimización de rutas")
st.markdown("Carga los archivos de red e instancia para calcular la **matriz de distancias** entre el depósito y los nodos de entrega.")

st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ── Carga de archivos ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### Archivo de red")
    st.caption("Formato: `NodoOrigen NodoDestino Distancia [flag]` por línea (Red1.nf)")
    net_file = st.file_uploader("Subir Red1.nf", type=["nf", "txt", "dat"],
                                 key="net", label_visibility="collapsed")

with col_right:
    st.markdown("### Instancia individual")
    st.caption("Formato compacto: `i 4 o 4 n 7 r … s …` (Individual_2.txt)")
    ind_file = st.file_uploader("Subir Individual_2.txt", type=["txt", "dat"],
                                 key="ind", label_visibility="collapsed")

# ── Estado de sesión ──────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = None
if "delivery_nodes" not in st.session_state:
    st.session_state.delivery_nodes = []
if "arcos_r" not in st.session_state:
    st.session_state.arcos_r = []
if "arcos_s" not in st.session_state:
    st.session_state.arcos_s = []
if "matrix" not in st.session_state:
    st.session_state.matrix = {}
if "paths" not in st.session_state:
    st.session_state.paths = {}

# ── Parsear archivos cuando se suban ─────────────────────────────────────────
if net_file:
    content = net_file.read().decode("utf-8", errors="ignore")
    graph, arc_count = parse_network_file(content)
    st.session_state.graph = graph
    st.session_state.arc_count = arc_count
    st.session_state.node_count = len(graph)

if ind_file:
    content = ind_file.read().decode("utf-8", errors="ignore")
    i_val, o_val, n_val, arcos_r, arcos_s = parse_individual_file(content)
    delivery_nodes = extract_delivery_nodes(arcos_s)
    st.session_state.delivery_nodes = delivery_nodes
    st.session_state.arcos_r = arcos_r
    st.session_state.arcos_s = arcos_s
    st.session_state.i_val = i_val
    st.session_state.o_val = o_val
    st.session_state.n_val = n_val

# ── Resumen de datos cargados ─────────────────────────────────────────────────
if st.session_state.graph or st.session_state.delivery_nodes:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Datos cargados")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        v = st.session_state.node_count if st.session_state.graph else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Nodos en red</div><div class="value">{v}</div></div>', unsafe_allow_html=True)
    with m2:
        v = st.session_state.arc_count if st.session_state.graph else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Arcos en red</div><div class="value">{v}</div></div>', unsafe_allow_html=True)
    with m3:
        v = len(st.session_state.delivery_nodes) if st.session_state.delivery_nodes else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Nodos entrega</div><div class="value">{v}</div><div class="sub">+ depósito (0)</div></div>', unsafe_allow_html=True)
    with m4:
        nodes = st.session_state.delivery_nodes
        tags = " ".join(f'<span class="tag">{n}</span>' for n in nodes) if nodes else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Nodos a visitar</div><div class="value" style="font-size:1rem;margin-top:4px">{tags}</div></div>', unsafe_allow_html=True)

# ── Visualización de red pequeña ──────────────────────────────────────────────
if st.session_state.arcos_r or st.session_state.arcos_s:
    with st.expander("📊 Ver red de servicio (Individual_2.txt)", expanded=False):
        fig = draw_small_network(
            st.session_state.arcos_r,
            st.session_state.arcos_s,
            st.session_state.delivery_nodes,
        )
        st.pyplot(fig, use_container_width=True)

# ── Botón de cálculo ──────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

ready = st.session_state.graph is not None and len(st.session_state.delivery_nodes) > 0
if not ready:
    st.info("Sube ambos archivos para habilitar el cálculo.")

calc_col, _ = st.columns([1, 3])
with calc_col:
    run = st.button("▶ Calcular matriz de distancias", disabled=not ready, use_container_width=True)

if run and ready:
    graph = st.session_state.graph
    delivery_nodes = st.session_state.delivery_nodes
    depot = 0
    all_points = [depot] + delivery_nodes

    matrix = {}
    paths = {}
    progress = st.progress(0, text="Ejecutando Dijkstra…")
    total = len(all_points)
    t0 = time.time()

    for idx, src in enumerate(all_points):
        dist_map, prev_map = dijkstra(graph, src)
        for dst in all_points:
            if src == dst:
                matrix[(src, dst)] = 0
                paths[(src, dst)] = [src]
            else:
                d = dist_map.get(dst, None)
                matrix[(src, dst)] = d
                paths[(src, dst)] = reconstruct_path(prev_map, src, dst) if d is not None else []
        progress.progress((idx + 1) / total,
                          text=f"Dijkstra desde nodo {src} ({idx+1}/{total})…")

    elapsed = time.time() - t0
    st.session_state.matrix = matrix
    st.session_state.paths = paths
    st.session_state.elapsed = elapsed
    progress.empty()
    st.success(f"✓ Matriz calculada en {elapsed:.3f} s")


# ── Resultados ────────────────────────────────────────────────────────────────
if st.session_state.matrix:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Resultados")

    matrix = st.session_state.matrix
    paths = st.session_state.paths
    delivery_nodes = st.session_state.delivery_nodes
    depot = 0
    all_points = [depot] + delivery_nodes
    labels = [f"Depósito (0)" if n == 0 else f"Nodo {n}" for n in all_points]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Matriz D[i,j]",
        "🛤 Rutas detalladas",
        "📄 Código AMPL",
        "💾 Exportar",
    ])

    # ── Tab 1: Matriz ─────────────────────────────────────────────────────────
    with tab1:
        st.caption("Distancia mínima entre cada par de nodos (depósito + entregas). Usá esta matriz como entrada del TSP.")
        df = display_matrix(matrix, all_points)
        df.index = labels
        df.columns = [f"→ {l}" for l in labels]
        st.dataframe(df, use_container_width=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("**Detalle numérico**")
        rows = []
        for i in all_points:
            for j in all_points:
                if i != j:
                    v = matrix.get((i, j))
                    rows.append({
                        "Desde": f"{'Depósito' if i==0 else 'Nodo'} {i}",
                        "Hasta": f"{'Depósito' if j==0 else 'Nodo'} {j}",
                        "Distancia": f"{v:,.1f}" if v is not None else "∞",
                        "Saltos": len(paths.get((i,j),[])) - 1,
                    })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=300)

    # ── Tab 2: Rutas ──────────────────────────────────────────────────────────
    with tab2:
        st.caption("Secuencia completa de nodos en el camino más corto entre cada par.")
        for i in all_points:
            for j in all_points:
                if i != j:
                    path = paths.get((i, j), [])
                    dist = matrix.get((i, j))
                    i_label = "Depósito" if i == 0 else f"Nodo {i}"
                    j_label = "Depósito" if j == 0 else f"Nodo {j}"
                    st.markdown(f'<div class="pair-header">{i_label} → {j_label} &nbsp;|&nbsp; distancia: <code>{dist:,.1f}</code> &nbsp;|&nbsp; {len(path)-1} arcos</div>', unsafe_allow_html=True)
                    if path:
                        arrow_path = " → ".join(str(n) for n in path)
                        st.markdown(f'<div class="path-row">{arrow_path}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Sin camino disponible.")

    # ── Tab 3: AMPL ───────────────────────────────────────────────────────────
    with tab3:
        st.caption("Código AMPL equivalente al cálculo realizado. Usá estos archivos para documentar el entregable.")

        st.markdown("**shortest_path.mod**")
        st.code(generate_ampl_mod(), language="text")

        st.markdown("**shortest_path.run** — resuelve todos los pares")
        st.code(generate_run_script(delivery_nodes), language="text")

        st.markdown("**Fragmento .dat generado** (la red completa viene de Red1.nf)")
        st.code(f"""\
# shortest_path.dat

set NODOS := ... ;  # todos los nodos de Red1.nf

set ARCOS := ... ;  # todos los arcos de Red1.nf

param dist := ... ;  # distancias de Red1.nf

# Cambiar para cada par:
param origen := 0 ;
param destino := {delivery_nodes[0] if delivery_nodes else 1} ;
""", language="text")

        st.markdown("**Bloque param D para TSP .dat** — pegá esto directamente en tu tsp.dat")
        lines = ["param D :"]
        header = "  " + "  ".join(str(p) for p in all_points)
        lines.append(header)
        for i in all_points:
            row = f"  {i}  " + "  ".join(
                "0" if i == j else (f"{matrix[(i,j)]:.1f}" if matrix.get((i,j)) is not None else "9999999")
                for j in all_points
            )
            lines.append(row)
        lines.append("  ;")
        st.code("\n".join(lines), language="text")

    # ── Tab 4: Exportar ───────────────────────────────────────────────────────
    with tab4:
        st.caption("Descargá los resultados para incluirlos en el informe técnico.")

        # CSV matriz
        csv_rows = []
        for i in all_points:
            for j in all_points:
                v = matrix.get((i, j))
                path = paths.get((i, j), [])
                csv_rows.append({
                    "desde": i,
                    "hasta": j,
                    "distancia": v if v is not None else "",
                    "ruta": " > ".join(str(x) for x in path),
                    "saltos": len(path) - 1 if path else "",
                })
        df_csv = pd.DataFrame(csv_rows)
        st.download_button(
            "⬇ Descargar resultados CSV",
            df_csv.to_csv(index=False).encode("utf-8"),
            file_name="matriz_distancias.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("")

        # .dat TSP
        lines = ["param D :", "  " + "  ".join(str(p) for p in all_points)]
        for i in all_points:
            row = f"  {i}  " + "  ".join(
                "0" if i == j else (f"{matrix[(i,j)]:.1f}" if matrix.get((i,j)) is not None else "9999999")
                for j in all_points
            )
            lines.append(row)
        lines.append("  ;")
        st.download_button(
            "⬇ Descargar bloque param D (.dat para TSP)",
            "\n".join(lines).encode("utf-8"),
            file_name="param_D_tsp.dat",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("")

        # AMPL .mod
        st.download_button(
            "⬇ Descargar shortest_path.mod",
            generate_ampl_mod().encode("utf-8"),
            file_name="shortest_path.mod",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("")
        st.download_button(
            "⬇ Descargar shortest_path.run",
            generate_run_script(delivery_nodes).encode("utf-8"),
            file_name="shortest_path.run",
            mime="text/plain",
            use_container_width=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.caption("II-1122 Optimización Industrial · Universidad de Costa Rica · Dijkstra (heap) O((V+E) log V)")

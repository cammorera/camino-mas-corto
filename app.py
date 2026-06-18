import streamlit as st
import heapq
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

st.set_page_config(
    page_title="Camino más corto — Redes",
    page_icon="🗺️",
    layout="wide",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding: 2rem 2.5rem 3rem; max-width: 1280px; }

h1 { font-size: 1.6rem !important; font-weight: 600 !important; letter-spacing: -0.02em; color: #0f172a; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1e293b; margin-top: 0 !important; }
h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #334155; }

.tag {
    display: inline-block; background: #f1f5f9; border: 1px solid #e2e8f0;
    border-radius: 4px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 2px 8px; color: #475569; margin: 2px 3px;
}
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.72rem; font-weight: 500; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-card .value { font-size: 1.5rem; font-weight: 600; color: #0f172a; font-family: 'JetBrains Mono', monospace; }
.metric-card .sub   { font-size: 0.78rem; color: #64748b; margin-top: 2px; }

.path-row {
    background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px;
    padding: 0.6rem 1rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; color: #166534; margin-bottom: 0.4rem;
    word-break: break-all;
}
.pair-header { font-size: 0.85rem; font-weight: 600; color: #1e293b; margin: 0.7rem 0 0.3rem; }

.section-rule { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0 1rem; }

.stAlert { border-radius: 6px !important; }
.stButton > button {
    background: #0f172a !important; color: white !important; border: none !important;
    border-radius: 6px !important; font-weight: 500 !important; font-size: 0.88rem !important;
    padding: 0.5rem 1.5rem !important; transition: background 0.2s;
}
.stButton > button:hover { background: #1e3a5f !important; }
.stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Parsers ───────────────────────────────────────────────────────────────────
def parse_network_file(content: str):
    """Parse Red1.nf: NodoOrigen NodoDestino Distancia [flag]"""
    graph = {}
    arc_count = 0
    for line in content.strip().splitlines()[1:]:   # skip header
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            u, v, d = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        graph.setdefault(u, []).append((v, d))
        arc_count += 1
    return graph, arc_count


def parse_clients_file(content: str):
    """Parse Clientes.txt: one node ID per line. Depot is always 0."""
    nodes = []
    for line in content.strip().splitlines():
        line = line.strip()
        if line.isdigit():
            nodes.append(int(line))
    return sorted(set(nodes))


# ── Dijkstra ──────────────────────────────────────────────────────────────────
def dijkstra(graph, source):
    dist = {source: 0}
    prev = {source: None}
    pq   = [(0, source)]
    visited = set()
    while pq:
        cost, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, w in graph.get(u, []):
            nc = cost + w
            if v not in dist or nc < dist[v]:
                dist[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    return dist, prev


def reconstruct_path(prev, source, target):
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return path if path and path[0] == source else []


# ── Helpers de display ────────────────────────────────────────────────────────
def display_matrix(matrix, nodes):
    data = {}
    for j in nodes:
        col = []
        for i in nodes:
            v = matrix.get((i, j))
            col.append("0" if i == j else ("∞" if v is None else f"{v:,.1f}"))
        data[str(j)] = col
    df = pd.DataFrame(data, index=[str(n) for n in nodes])
    df.index.name = "Desde \\ Hasta"
    return df


# ── Generadores AMPL ──────────────────────────────────────────────────────────
def generate_ampl_mod():
    return """\
# shortest_path.mod
# Modelo de flujo mínimo para camino más corto (un par a la vez)

set NODOS;
set ARCOS within NODOS cross NODOS;

param dist   {ARCOS} >= 0;
param origen symbolic in NODOS;
param destino symbolic in NODOS;

var f {ARCOS} >= 0;

minimize total_dist:
    sum {(i,j) in ARCOS} dist[i,j] * f[i,j];

s.t. balance {k in NODOS}:
    sum {(k,j) in ARCOS} f[k,j]
  - sum {(i,k) in ARCOS} f[i,k]
  = if k = origen  then  1
    else if k = destino then -1
    else 0;
"""


def generate_run_script(delivery_nodes, depot=0):
    all_points = [depot] + delivery_nodes
    pairs = [(i, j) for i in all_points for j in all_points if i != j]
    lines = [
        "# shortest_path.run",
        "model shortest_path.mod;",
        "data  shortest_path.dat;",
        "",
    ]
    for orig, dest in pairs:
        lines += [
            f"let origen  := {orig};",
            f"let destino := {dest};",
            "solve;",
            f'printf "D[{orig}][{dest}] = %g\\n", total_dist;',
            "",
        ]
    return "\n".join(lines)


def generate_param_D_block(matrix, all_points):
    lines = ["param D :"]
    lines.append("  " + "  ".join(str(p) for p in all_points))
    for i in all_points:
        row_vals = []
        for j in all_points:
            if i == j:
                row_vals.append("0")
            else:
                v = matrix.get((i, j))
                row_vals.append(f"{v:.1f}" if v is not None else "9999999")
        lines.append(f"  {i}  " + "  ".join(row_vals))
    lines.append("  ;")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🗺️ Camino más corto — Optimización de rutas")
st.markdown("Carga la red vial y el archivo de clientes para calcular la **matriz de distancias mínimas** D[i,j].")
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ── Carga de archivos ─────────────────────────────────────────────────────────
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown("### Archivo de red")
    st.caption("Formato: `NodoOrigen NodoDestino Distancia` por línea (ej. Red1.nf)")
    net_file = st.file_uploader("Red1.nf", type=["nf", "txt", "dat"],
                                 key="net", label_visibility="collapsed")

with col_r:
    st.markdown("### Archivo de clientes")
    st.caption("Formato: un nodo por línea (ej. Clientes1.txt). El depósito siempre es el nodo 0.")
    cli_file = st.file_uploader("Clientes1.txt", type=["txt", "dat"],
                                 key="cli", label_visibility="collapsed")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("graph", None), ("arc_count", 0), ("node_count", 0),
    ("delivery_nodes", []), ("matrix", {}), ("paths", {}), ("elapsed", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Parse on upload ───────────────────────────────────────────────────────────
if net_file:
    content = net_file.read().decode("utf-8", errors="ignore")
    graph, arc_count = parse_network_file(content)
    st.session_state.graph      = graph
    st.session_state.arc_count  = arc_count
    st.session_state.node_count = len(graph)

if cli_file:
    content = cli_file.read().decode("utf-8", errors="ignore")
    delivery_nodes = parse_clients_file(content)
    st.session_state.delivery_nodes = delivery_nodes

# ── Resumen de datos ──────────────────────────────────────────────────────────
if st.session_state.graph or st.session_state.delivery_nodes:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Datos cargados")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        v = st.session_state.node_count if st.session_state.graph else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Nodos en red</div><div class="value">{v}</div></div>', unsafe_allow_html=True)
    with m2:
        v = f"{st.session_state.arc_count:,}" if st.session_state.graph else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Arcos en red</div><div class="value">{v}</div></div>', unsafe_allow_html=True)
    with m3:
        n = len(st.session_state.delivery_nodes)
        v = n if n else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Clientes</div><div class="value">{v}</div><div class="sub">+ depósito (nodo 0)</div></div>', unsafe_allow_html=True)
    with m4:
        nodes = st.session_state.delivery_nodes
        # show first 12 tags to avoid overflow
        shown = nodes[:12]
        tags  = " ".join(f'<span class="tag">{n}</span>' for n in shown)
        extra = f'<span class="tag">+{len(nodes)-12} más</span>' if len(nodes) > 12 else ""
        val   = (tags + extra) if nodes else "—"
        st.markdown(f'<div class="metric-card"><div class="label">Nodos a visitar</div><div style="margin-top:6px;line-height:2">{val}</div></div>', unsafe_allow_html=True)

# ── Advertencia tamaño ────────────────────────────────────────────────────────
n_clients = len(st.session_state.delivery_nodes)
if n_clients > 0:
    n_pairs = (n_clients + 1) * n_clients   # N+1 nodos → N*(N+1) pares dirigidos
    st.info(
        f"Se calcularán **{n_clients + 1} corridas de Dijkstra** "
        f"(depósito + {n_clients} clientes) → **{n_pairs} distancias** en la matriz.",
        icon="ℹ️",
    )

# ── Botón calcular ────────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

ready = st.session_state.graph is not None and n_clients > 0
if not ready:
    st.info("Sube ambos archivos para habilitar el cálculo.")

calc_col, _ = st.columns([1, 3])
with calc_col:
    run = st.button("▶ Calcular matriz de distancias", disabled=not ready, use_container_width=True)

if run and ready:
    graph          = st.session_state.graph
    delivery_nodes = st.session_state.delivery_nodes
    depot          = 0
    all_points     = [depot] + delivery_nodes

    matrix, paths = {}, {}
    progress = st.progress(0, text="Ejecutando Dijkstra…")
    t0 = time.time()

    for idx, src in enumerate(all_points):
        dist_map, prev_map = dijkstra(graph, src)
        for dst in all_points:
            if src == dst:
                matrix[(src, dst)] = 0
                paths[(src, dst)]  = [src]
            else:
                d = dist_map.get(dst)
                matrix[(src, dst)] = d
                paths[(src, dst)]  = reconstruct_path(prev_map, src, dst) if d is not None else []
        progress.progress(
            (idx + 1) / len(all_points),
            text=f"Dijkstra desde nodo {src}  ({idx+1}/{len(all_points)})…"
        )

    elapsed = time.time() - t0
    st.session_state.matrix  = matrix
    st.session_state.paths   = paths
    st.session_state.elapsed = elapsed
    progress.empty()
    st.success(f"✓ Matriz calculada en {elapsed:.3f} s")


# ── Resultados ────────────────────────────────────────────────────────────────
if st.session_state.matrix:
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("### Resultados")

    matrix         = st.session_state.matrix
    paths          = st.session_state.paths
    delivery_nodes = st.session_state.delivery_nodes
    depot          = 0
    all_points     = [depot] + delivery_nodes

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Matriz D[i,j]",
        "🛤 Rutas detalladas",
        "📄 Código AMPL",
        "💾 Exportar",
    ])

    # ── Tab 1: Matriz ─────────────────────────────────────────────────────────
    with tab1:
        st.caption("Distancia mínima entre cada par. Fila = origen, columna = destino. Usá esta matriz como entrada del TSP.")

        df = display_matrix(matrix, all_points)
        st.dataframe(df, use_container_width=True, height=min(600, 40 + 35 * len(all_points)))

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("**Vista de lista**")
        rows = []
        for i in all_points:
            for j in all_points:
                if i != j:
                    v = matrix.get((i, j))
                    rows.append({
                        "Desde": i,
                        "Hasta": j,
                        "Distancia": round(v, 1) if v is not None else None,
                        "Saltos":    len(paths.get((i, j), [])) - 1,
                    })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

    # ── Tab 2: Rutas detalladas ───────────────────────────────────────────────
    with tab2:
        st.caption("Secuencia completa de nodos en el camino más corto. Para instancias grandes, filtrá por nodo de origen.")

        # Filtro por origen
        origin_options = ["Todos"] + [str(p) for p in all_points]
        sel = st.selectbox("Mostrar rutas desde el nodo:", origin_options, key="route_filter")

        shown_origins = all_points if sel == "Todos" else [int(sel)]

        for i in shown_origins:
            for j in all_points:
                if i == j:
                    continue
                path = paths.get((i, j), [])
                dist = matrix.get((i, j))
                i_lbl = "Depósito (0)" if i == 0 else f"Nodo {i}"
                j_lbl = "Depósito (0)" if j == 0 else f"Nodo {j}"
                dist_str = f"{dist:,.1f}" if dist is not None else "∞"
                hops = len(path) - 1 if path else "—"
                st.markdown(
                    f'<div class="pair-header">{i_lbl} → {j_lbl} &nbsp;|&nbsp; '
                    f'dist: <code>{dist_str}</code> &nbsp;|&nbsp; {hops} arcos</div>',
                    unsafe_allow_html=True,
                )
                if path:
                    arrow_path = " → ".join(str(n) for n in path)
                    st.markdown(f'<div class="path-row">{arrow_path}</div>', unsafe_allow_html=True)
                else:
                    st.warning("Sin camino disponible.")

    # ── Tab 3: AMPL ───────────────────────────────────────────────────────────
    with tab3:
        st.caption("Archivos AMPL listos para copiar o descargar.")

        st.markdown("**shortest_path.mod**")
        st.code(generate_ampl_mod(), language="text")

        st.markdown("**shortest_path.run** — itera sobre todos los pares")
        st.code(generate_run_script(delivery_nodes), language="text")

        st.markdown("**Bloque `param D` para tsp.dat** — pegá esto directo en tu archivo de datos del TSP")
        st.code(generate_param_D_block(matrix, all_points), language="text")

    # ── Tab 4: Exportar ───────────────────────────────────────────────────────
    with tab4:
        st.caption("Descargá los archivos para el informe técnico y el modelo TSP.")

        # CSV
        csv_rows = []
        for i in all_points:
            for j in all_points:
                v    = matrix.get((i, j))
                path = paths.get((i, j), [])
                csv_rows.append({
                    "desde":     i,
                    "hasta":     j,
                    "distancia": round(v, 1) if v is not None else "",
                    "ruta":      " > ".join(str(x) for x in path),
                    "saltos":    len(path) - 1 if path else "",
                })
        st.download_button(
            "⬇ Descargar resultados (CSV)",
            pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8"),
            file_name="matriz_distancias.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("")
        st.download_button(
            "⬇ Descargar param D — bloque .dat para TSP",
            generate_param_D_block(matrix, all_points).encode("utf-8"),
            file_name="param_D_tsp.dat",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("")
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

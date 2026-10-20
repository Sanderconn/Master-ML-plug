import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Graph Viewer", layout="wide")

st.title("Graph Viewer")
st.markdown("---")

SCENARIO_DESCRIPTIONS = {
    0: {
        "name": "Normal operation",
        "description": """
        Scenario 0 represents normal and healthy well operation. 
        """,
    },
    1: {
        "name": "Abrupt increase of BSW",
        "description": """
        Scenario 1 represents an abrupt increase in basic sediment and water content in the produced fluid.
        This can affect the pressure readings as the well may experience a change in flow regime or increased backpressure.
        """,
    },
    2: {
        "name": "Spurious closure of DHSV",
        "description": """
        Scenario 2 represents a spurious closure of the downhole safety valve.
        Meaning that the safety valve starts closing unexpectedly, despite normal well operation.
        """,
    },
    3: {
        "name": "Severe slugging",
        "description": """
        Scenario 3 represents severe slugging, a multiphase-flow instability that can
        cause large oscillations in pressure and flow.
        """,
    },
    4: {
        "name": "Flow instability",
        "description": """
        Scenario 4 represents unstable flow behavior. Usually not periodic like severe slugging.
        Amplitudes and severity of the sensor readings are more tolerable, but still abnormal.
        """,
    },
    5: {
        "name": "Rapid productivity loss",
        "description": """
        Scenario 5 represents rapid productivity loss in the well.
        The productivity of a naturally flowing well depends on several
        properties. These properties can change, causing the well to lose its ability to produce as the flow slows or even stops.
        """,
    },
    6: {
        "name": "Quick restriction in PCK",
        "description": """
        Scenario 6 represents a quick restriction in the production choke. 
        The well is flowing normally, then the surface control valve closes suddenly, causing an abrupt increase in resistance to flow
        """,
    },
    7: {
        "name": "Scaling in PCK",
        "description": """
        Scenario 7 represents scaling in the production choke. 
        Mineral deposits gradually build up in the production choke, progressively restricting the flow.
        """,
    },
    8: {
        "name": "Hydrate in production line",
        "description": """
        Scenario 8 represents hydrate formation in the production line. 
        Solid deposits are formed, altering and restricting flow.
        """,
    },
}

GRAPH_ORDER = [
    "_predicted_label.png",
    "_actual_label.png",
    "_anomaly_score.png",
    "_relative_ratio_deviation.png",
]

def graph_sort_key(path):
    name = path.name

    for index, suffix in enumerate(GRAPH_ORDER):
        if name.endswith(suffix):
            return index


    return len(GRAPH_ORDER)

# --- Sidebar controls ---
st.sidebar.header("Navigation")

scenario = st.sidebar.selectbox(
    "Select Scenario",
    options=list(range(9)),
    format_func=lambda x: f"Scenario {x}",
)

run = st.sidebar.selectbox(
    "Select Run",
    options=["run1", "run2", "run3", "run4"],
)

# --- Resolve graph folder ---
base_dir = Path(__file__).parent
folder = base_dir / str(scenario) / run

scenario_info = SCENARIO_DESCRIPTIONS.get(
    scenario,
    {
        "name": f"Scenario {scenario}",
        "description": "No description available for this scenario.",
    },
)

st.subheader(f"Scenario {scenario}: {scenario_info['name']} — {run}")

with st.expander("Scenario description", expanded=True):
    st.markdown(scenario_info["description"])

if not folder.exists():
    st.warning(f"Folder not found: `{folder}`")
else:
    graphs = sorted(folder.glob("*.png"), key=graph_sort_key)

    if not graphs:
        st.info("No PNG files found in this folder.")
    else:
        # Display up to 4 graphs in a 2×2 grid
        cols_per_row = 1
        for i in range(0, len(graphs), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(graphs):
                    img_path = graphs[i + j]
                    with col:
                        st.image(
                            Image.open(img_path),
                            caption=img_path.stem,
                            width='stretch',
                        )
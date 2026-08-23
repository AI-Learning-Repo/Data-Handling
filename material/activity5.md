# Laboratory Guide: Building Interactive Web Applications with Streamlit in Google Colab

---

## 1. Overview and Objectives

Streamlit is an open-source Python framework designed for creating interactive web applications, data dashboards, and machine learning interfaces without requiring frontend web development experience (HTML/CSS/JavaScript).

### Learning Objectives
By completing this laboratory, you will:
1. Configure and host a live Streamlit server directly inside Google Colab using Google's native port proxy (`google.colab.output`).
2. Understand Streamlit's reactive execution model and widget submission mechanics.
3. Build user interfaces using input widgets, interactive tables, and KPI metric cards.
4. Render static and interactive visualizations across multiple plotting frameworks (Matplotlib, Seaborn, Plotly, Native Streamlit).
5. Manage persistent application state across reruns using `st.session_state`.
6. Optimize computational performance using data caching (`@st.cache_data`).
7. Construct modular dashboard layouts using sidebars, columns, expanders, and tabs.
8. Control execution flow with batch forms and progress indicators.

---

## 2. Architecture & Execution Model

### 2.1 The Streamlit Execution Model
Streamlit uses a **reactive top-to-bottom execution cycle**:
* Whenever an end-user interacts with an active widget (e.g., dragging a slider or clicking a button), Streamlit executes the entire Python script from top to bottom.
* Widgets directly return their current state to the Python variables they are assigned to.
* **Input Commitment Rules:**
  * Sliders, radio buttons, checkboxes, and select boxes trigger reruns **immediately** on change.
  * Text inputs require the user to press **`Enter`** (or unfocus the text field) before committing the value and executing the script.

### 2.2 Google Colab Native Port Proxy
Rather than relying on third-party tunnels that can fail on dynamic JavaScript modules or trigger antivirus blocks, this lab uses **Google Colab's native kernel proxy**:

```
+-------------------------------------------------------------+
| Google Colab Linux Container                                |
|                                                             |
|   +--------------------------+                              |
|   | app.py (Streamlit Server)|                              |
|   | Listening on Port :8501  |                              |
|   +------------+-------------+                              |
+----------------|--------------------------------------------+
                 | Native Kernel Bridge
                 v
+-------------------------------------------------------------+
| Google Cloud Infrastructure (*.colab.googleusercontent.com) |
|                                                             |
|   [ Option A: Embedded In-Notebook IFrame View ]            |
|   [ Option B: Full-Screen Browser Tab View     ]            |
+-------------------------------------------------------------+
```

---

## 3. Environment Setup & Application Launch

Execute the following cell in your Google Colab notebook to install dependencies, launch the background server, and render the interactive interface.

```python
# 1. Install Streamlit and visualization libraries (fast pre-built binary wheels)
!pip install -q streamlit plotly seaborn matplotlib

# 2. Write baseline application
with open("app.py", "w") as f:
    f.write("""import streamlit as st
st.set_page_config(page_title="Streamlit Colab Lab", layout="centered")
st.title("Streamlit Environment Online")
st.success("Server initialized successfully. Proceed to Module 1 below.")
""")

# 3. Terminate any previous instances cleanly
import subprocess
import time
subprocess.run(["pkill", "-f", "streamlit"], stderr=subprocess.DEVNULL)

# 4. Launch Streamlit server in the background
subprocess.Popen([
    "streamlit", "run", "app.py",
    "--server.port", "8501",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Allow server time to bind port
time.sleep(3)

# 5. Generate direct URL and display embedded iframe
from google.colab import output
from google.colab.output import eval_js

proxy_url = eval_js("google.colab.kernel.proxyPort(8501)")

print("=====================================================================")
print(f"FULL-SCREEN DIRECT URL (OPTIONAL): {proxy_url}")
print("=====================================================================")

# Render interactive iframe in notebook cell output:
output.serve_kernel_port_as_iframe(8501, height='650')
```

### How to Interact with Modules in this Lab
1. **To update the app:** Run any of the module cells below in Colab (each uses `%%writefile app.py` to update the script).
2. **To view changes:** 
   * **In the Notebook:** Scroll up to the setup cell's iframe, click inside it, and press **`R`** on your keyboard (or click **Menu** $\rightarrow$ **Rerun** in the top right of the iframe).
   * **In a Separate Tab:** Open the printed `FULL-SCREEN DIRECT URL` and refresh the page.

---

## 4. Hands-On Laboratory Modules

---

### Module 1: Basic Input Widgets

This module introduces fundamental interactive input components.

```python
%%writefile app.py
import streamlit as st

st.set_page_config(page_title="Module 1: Widgets", layout="centered")
st.title("Module 1: Basic Input Widgets")

st.info("Remember: Press Enter after typing in the text input to commit your change.")

# Text input
name = st.text_input("Enter your name:", placeholder="Type name and press Enter...")
if name:
    st.success(f"Hello, {name}!")

# Numerical input
age = st.number_input("Enter your age:", min_value=0, max_value=120, value=25, step=1)
st.write(f"Registered age: **{age}**")

# Slider
score = st.slider("Select an evaluation score:", min_value=1, max_value=10, value=5)
st.write(f"Selected score: **{score}**")

# Button trigger
if st.button("Execute Action"):
    st.info("Action button triggered.")

# Checkbox toggle
if st.checkbox("Show technical details"):
    st.write("Additional technical metadata displayed conditionally.")

# Categorical options
color = st.radio("Select primary color:", ["Red", "Green", "Blue"])
st.write(f"Selected color: **{color}**")

option = st.selectbox("Select discrete category:", [f"Category {i}" for i in range(1, 6)])
st.write(f"Selected category: **{option}**")

langs = st.multiselect(
    "Select technical proficiencies:", 
    ["Python", "R", "SQL", "C++", "Julia"]
)
st.write(f"Proficiencies: **{', '.join(langs) if langs else 'None selected'}**")
```

#### Concepts Explained
* **Widget Return Values:** Widgets return their value directly into variables without callback boilerplate.
* **Reactive Model:** Any interaction re-evaluates the entire script to reflect the latest state.

---

### Module 2: DataFrames, Tables, and Metrics

This module covers interactive tabular displays, cell editing, and KPI cards.

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Module 2: Data Display", layout="wide")
st.title("Module 2: Data Display & Metrics")

np.random.seed(42)
df = pd.DataFrame(
    np.random.randn(8, 4),
    columns=["Metric A", "Metric B", "Metric C", "Metric D"]
)

st.subheader("1. Interactive DataFrame (`st.dataframe`)")
st.caption("Scrollable, sortable, and resizable table representation.")
st.dataframe(df, use_container_width=True)

st.subheader("2. Interactive Data Editor (`st.data_editor`)")
st.caption("Allows direct cell editing. Double-click any cell to modify values.")
edited_df = st.data_editor(df, use_container_width=True)

st.subheader("3. Static HTML Table (`st.table`)")
st.caption("Fixed presentation suitable for small summary matrices.")
st.table(df.head(3))

st.subheader("4. Metric Cards (`st.metric`)")
col1, col2, col3 = st.columns(3)
col1.metric("Server Throughput", "1,240 req/s", "45 req/s")
col2.metric("Response Latency", "42 ms", "-4 ms", delta_color="inverse")
col3.metric("Error Rate", "0.02%", "0.00%")
```

#### Concepts Explained
* `st.dataframe`: Renders an interactive virtualized table that supports sorting and column resizing.
* `st.data_editor`: Allows in-place cell editing and returns the mutated DataFrame back to Python.
* `st.metric`: Standard UI component for displaying numerical values along with positive/negative delta indicators.

---

### Module 3: Data Visualizations (Static, Interactive, and Native)

This module demonstrates plotting via Matplotlib, Seaborn, Plotly Express, and native chart wrappers.

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Module 3: Visualizations", layout="wide")
st.title("Module 3: Visualization Frameworks")

np.random.seed(42)
data = pd.DataFrame({
    "Step": np.arange(100),
    "Series_1": np.random.randn(100).cumsum(),
    "Series_2": np.random.randn(100).cumsum(),
    "Category": np.random.choice(["Type A", "Type B", "Type C"], 100)
})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Matplotlib (Static)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["Step"], data["Series_1"], color="steelblue", label="Series 1")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Seaborn (Statistical)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=data, x="Series_1", y="Series_2", hue="Category", ax=ax)
    st.pyplot(fig)

st.divider()

st.subheader("Plotly (Interactive Client-Side)")
fig_px = px.scatter(
    data, 
    x="Series_1", 
    y="Series_2", 
    color="Category", 
    title="Interactive Scatter Plot (Supports Zoom, Pan, and Hover Tooltips)"
)
st.plotly_chart(fig_px, use_container_width=True)

st.divider()

st.subheader("Native Streamlit Charts")
st.line_chart(data[["Series_1", "Series_2"]], use_container_width=True)
st.bar_chart(data.groupby("Category")["Series_1"].mean(), use_container_width=True)
```

#### Concepts Explained
* Static plots (`st.pyplot`) are rendered on the server and sent as static images.
* Interactive plots (`st.plotly_chart`) send JSON definitions to the client browser, enabling hardware-accelerated tooltips, zooming, and panning.

---

### Module 4: Performance Caching (`@st.cache_data`)

This module demonstrates caching expensive computations to avoid blocking reruns.

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Module 4: Caching", layout="centered")
st.title("Module 4: Computation Caching")

@st.cache_data(ttl=60)
def execute_heavy_computation(n_rows: int) -> pd.DataFrame:
    # Simulate an expensive data transformation or database query
    time.sleep(3)
    return pd.DataFrame(
        np.random.randn(n_rows, 2), 
        columns=["Feature_X", "Feature_Y"]
    )

sample_size = st.slider("Select sample size (n):", 100, 2000, 500, step=100)

if st.button("Run Processing"):
    start_time = time.time()
    with st.spinner("Executing pipeline..."):
        result_df = execute_heavy_computation(sample_size)
    elapsed = time.time() - start_time
    
    st.success(f"Completed in {elapsed:.2f} seconds.")
    st.line_chart(result_df)
else:
    st.info("Adjust the slider and click the button to observe caching behavior.")
```

#### Concepts Explained
* `@st.cache_data`: Caches the return value of a function based on input arguments and function bytecode. Subsequent calls with matching arguments return the cached result instantly.
* `ttl=60`: Sets a Time-To-Live expiration of 60 seconds on cached data.

---

### Module 5: Session State Management (`st.session_state`)

This module demonstrates preserving and mutating state across application reruns.

```python
%%writefile app.py
import streamlit as st

st.set_page_config(page_title="Module 5: Session State", layout="centered")
st.title("Module 5: State Management")

# Initialize persistent session keys
if "counter" not in st.session_state:
    st.session_state.counter = 0

if "event_log" not in st.session_state:
    st.session_state.event_log = []

def increment():
    st.session_state.counter += 1
    st.session_state.event_log.append("Incremented")

def decrement():
    st.session_state.counter -= 1
    st.session_state.event_log.append("Decremented")

def reset():
    st.session_state.counter = 0
    st.session_state.event_log.append("Reset")

st.metric("Persistent Counter Value", st.session_state.counter)

col1, col2, col3 = st.columns(3)
with col1:
    st.button("Increment (+1)", on_click=increment, use_container_width=True)
with col2:
    st.button("Decrement (-1)", on_click=decrement, use_container_width=True)
with col3:
    st.button("Reset (0)", on_click=reset, use_container_width=True)

with st.expander("Session State Inspector"):
    st.json({
        "counter": st.session_state.counter,
        "event_log": st.session_state.event_log
    })
```

#### Concepts Explained
* `st.session_state`: A persistent dictionary that retains user variables across top-to-bottom script reruns for the lifetime of the session.
* Callbacks (`on_click`) execute state modifications before the main script evaluates.

---

### Module 6: UI Layouts and Modular Containers

This module covers dashboard organization with sidebars, columns, tabs, and expanders.

```python
%%writefile app.py
import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Module 6: Layouts", layout="wide")

# Sidebar Controls
st.sidebar.header("Global Configuration")
env = st.sidebar.selectbox("Environment", ["Development", "Staging", "Production"])
theme_color = st.sidebar.color_picker("Accent Color", "#1f77b4")
n_records = st.sidebar.slider("Number of records", 10, 100, 25)

st.title("Module 6: UI Layout Containers")
st.caption(f"Active Environment: **{env}**")

# Column Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Summary Panel")
    st.markdown(f"Selected Color: `{theme_color}`")
    st.write(f"Sample Size: **{n_records}**")

with col2:
    st.subheader("Data Output")
    synth_data = pd.DataFrame(np.random.randn(n_records, 2), columns=["Val_1", "Val_2"])
    st.line_chart(synth_data)

# Tabbed Layout
tab_data, tab_stats, tab_info = st.tabs(["Raw Data", "Descriptive Statistics", "Metadata"])

with tab_data:
    st.dataframe(synth_data, use_container_width=True)

with tab_stats:
    st.write(synth_data.describe())

with tab_info:
    with st.expander("Container Details"):
        st.write("Expanders and tabs organize visual information and prevent excessive vertical page scrolling.")
```

#### Concepts Explained
* `st.sidebar`: Positions widgets into a collapsible side navigation drawer.
* `st.columns`: Divides horizontal screen real estate using relative width weighting.
* `st.tabs` and `st.expander`: Provide layered interfaces to present complex data without clutter.

---

### Module 7: File Ingestion and Media Processing

This module demonstrates file uploading and embedding multimedia.

```python
%%writefile app.py
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Module 7: Files & Media", layout="wide")
st.title("Module 7: File Handling & Media")

uploaded_file = st.file_uploader(
    "Upload a tabular file (CSV) or an image (PNG/JPG)", 
    type=["csv", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded CSV Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image Preview")
        st.image(image, caption="Uploaded Asset", use_container_width=True)
else:
    st.info("Awaiting file upload.")

st.divider()

st.subheader("Media Stream Players")
c1, c2 = st.columns(2)
with c1:
    st.caption("Audio Player")
    st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3")
with c2:
    st.caption("Video Player")
    st.video("https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4")
```

#### Concepts Explained
* `st.file_uploader`: Reads uploaded files directly into in-memory byte buffers (`BytesIO`/`StringIO`) without requiring temporary disk writes.

---

### Module 8: Status Notifications and Progress Tracking

This module covers progress indicators, real-time workflow statuses, and alert banners.

```python
%%writefile app.py
import streamlit as st
import time

st.set_page_config(page_title="Module 8: Status", layout="centered")
st.title("Module 8: Progress & Status Indicators")

if st.button("Start Workflow"):
    with st.status("Executing pipeline workflow...", expanded=True) as status:
        st.write("Step 1: Ingesting data sources...")
        time.sleep(1)
        
        st.write("Step 2: Processing transformations...")
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)
            progress_bar.progress(i)
            
        st.write("Step 3: Validating schema...")
        time.sleep(0.5)
        
        status.update(label="Pipeline completed successfully.", state="complete", expanded=False)
        
    st.success("Execution completed.")
    st.balloons()
else:
    st.info("Click the button to start the multi-step process.")
```

#### Concepts Explained
* `st.status`: Provides a dynamic container that visibly transitions between `running`, `complete`, and `error` states.
* `st.progress`: Renders an integer-driven percentage completion bar.

---

### Module 9: Batch Execution with Forms (`st.form`)

This module demonstrates batching user inputs to suppress premature script reruns.

```python
%%writefile app.py
import streamlit as st

st.set_page_config(page_title="Module 9: Forms", layout="centered")
st.title("Module 9: Batch Processing (Forms)")

st.write("Widgets inside a form do not trigger script reruns until the submit button is clicked.")

with st.form(key="registration_form"):
    st.subheader("User Registration")
    user_id = st.text_input("User ID")
    role = st.selectbox("Role", ["Data Engineer", "Data Scientist", "ML Engineer", "Analyst"])
    access_level = st.slider("Access Level", 1, 5, 2)
    subscribe_alerts = st.checkbox("Subscribe to system alerts", value=True)
    
    # Explicit submit button
    submitted = st.form_submit_button("Submit Record")

if submitted:
    if not user_id.strip():
        st.error("Validation error: User ID cannot be empty.")
    else:
        st.success("Form submitted successfully.")
        st.json({
            "User_ID": user_id,
            "Role": role,
            "Access_Level": access_level,
            "Alerts_Enabled": subscribe_alerts
        })
```

#### Concepts Explained
* `st.form`: Decouples widget interactions from immediate server reruns, batching all form data into a single update cycle upon clicking `st.form_submit_button`.

---

## 5. Capstone Project: Integrated Data Analysis Dashboard

Combine all concepts covered in this lab into a unified analytics dashboard that explores the Iris dataset.

```python
%%writefile app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Iris Analysis Dashboard", layout="wide")

# Cached dataset loading
@st.cache_data
def load_dataset() -> pd.DataFrame:
    return sns.load_dataset("iris")

df = load_dataset()

# Sidebar controls
st.sidebar.header("Filter Configuration")
all_species = df["species"].unique().tolist()
selected_species = st.sidebar.multiselect(
    "Select Species:", 
    options=all_species, 
    default=all_species
)

# Apply filtering
filtered_df = df[df["species"].isin(selected_species)]

# Main Dashboard Title
st.title("Iris Dataset Exploration Dashboard")
st.markdown("Integrated analytics dashboard demonstrating widgets, caching, session layouts, and multi-engine plotting.")

# KPI Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(filtered_df))
col2.metric("Mean Sepal Length", f"{filtered_df['sepal_length'].mean():.2f} cm" if not filtered_df.empty else "N/A")
col3.metric("Mean Petal Length", f"{filtered_df['petal_length'].mean():.2f} cm" if not filtered_df.empty else "N/A")
col4.metric("Species Count", filtered_df["species"].nunique())

st.divider()

# Tabbed layout
tab_charts, tab_data, tab_summary = st.tabs(["Visualizations", "Data Table", "Statistical Summary"])

with tab_charts:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Feature Correlation (Plotly)")
        if not filtered_df.empty:
            fig_scatter = px.scatter(
                filtered_df, 
                x="sepal_length", 
                y="sepal_width", 
                color="species",
                size="petal_length",
                hover_data=["petal_width"],
                title="Sepal Dimensions vs. Petal Length"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No data matching current filters.")

    with c2:
        st.subheader("Distribution Analysis (Seaborn)")
        if not filtered_df.empty:
            fig_box, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=filtered_df, x="species", y="petal_length", ax=ax, palette="Set2")
            ax.set_ylabel("Petal Length (cm)")
            st.pyplot(fig_box)
        else:
            st.warning("No data matching current filters.")

with tab_data:
    st.subheader("Filtered Dataset View")
    st.dataframe(filtered_df, use_container_width=True)

with tab_summary:
    st.subheader("Descriptive Statistics")
    if not filtered_df.empty:
        st.write(filtered_df.describe())
    else:
        st.info("Select at least one species in the sidebar to view statistics.")
```

---

## 6. Summary Reference Table

| Feature Category | Core API Primitives | Functional Description |
| :--- | :--- | :--- |
| **Input Widgets** | `st.text_input`, `st.slider`, `st.button`, `st.selectbox`, `st.multiselect` | Captures typed inputs, numerical values, click events, and categorical choices. |
| **Data Display** | `st.dataframe`, `st.data_editor`, `st.table`, `st.metric` | Renders interactive tables, editable matrices, static tables, and KPI metric cards. |
| **Visualization** | `st.pyplot`, `st.plotly_chart`, `st.line_chart`, `st.bar_chart` | Renders static, interactive client-side, and native lightweight charts. |
| **Optimization** | `@st.cache_data`, `@st.cache_resource` | Persists function outputs in memory to prevent redundant computation across reruns. |
| **State Persistence** | `st.session_state` | Preserves mutable variables across top-to-bottom script execution cycles. |
| **Layout Controls** | `st.sidebar`, `st.columns`, `st.tabs`, `st.expander` | Controls horizontal and vertical layout organization across modular views. |
| **Execution Flow** | `st.form`, `st.form_submit_button`, `st.status`, `st.spinner` | Batches input events and displays progress notifications during long-running tasks. |
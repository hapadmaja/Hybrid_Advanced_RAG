import streamlit as st
import subprocess
import time
import psutil
import webbrowser

# Define programs with their respective ports, descriptions, and colors
PROGRAMS = {
    "🚀 Hybrid RAG": ("Hybrid_RAG.py", 8502, "This combines self corrective mechanism to reduce hallucinations", "#9fdbf4"),
    "🤖 Advanced RAG-based QA": ("Advanced_RAG_QA.py", 8503, "This uses dynamic query engine selection for improved responses.","#c4f4de"),
    "📄 Report Generation": ("Generate_Report.py", 8504, "Advanced RAG based Report generation", "	#fcd2d6")
}

# Function to stop a process running on a specific port
def kill_process_on_port(port):
    for proc in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            cmdline = " ".join(proc.info['cmdline']) if proc.info['cmdline'] else ""
            if f"--server.port {port}" in cmdline:
                proc.kill()
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

# Function to start a selected application
def start_selected_app(script, port):
    # Stop any previously running instance
    for _, existing_port, _, _ in PROGRAMS.values():
        if existing_port != port:
            kill_process_on_port(existing_port)

    # Start the selected program
    subprocess.Popen(
        ["streamlit", "run", script, "--server.port", str(port), "--server.headless", "true"],
        shell=True
    )
    time.sleep(3)  # Give time for the app to start

    # Open the application in a new browser tab
    webbrowser.open_new(f"http://localhost:{port}")

    st.success(f"Launching application at **http://localhost:{port}** in a new tab...")

# Apply custom styles
st.markdown(
    """
    <style>
        .title {
            font-size: 32px;
            text-align: center;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
        }
        .app-card {
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            align-items: center;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            height: 185px;  /* Ensure uniform height */
            width: 100%;  /* Full width within column */
            transition: all 0.3s ease-in-out;
        }
        .app-card:hover {
            transform: scale(1.05);
            box-shadow: 3px 3px 20px rgba(0, 0, 0, 0.15);
        }
        .app-description {
            font-size: 16px;
            color: #555;
            flex-grow: 1;
            display: flex;
            align-items: center;
            text-align: center;
            justify-content: center;
        }
        .styled-button {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            text-align: center;
            border-radius: 10px;
            color: white;
            background: linear-gradient(145deg, #007bff, #0056b3);
            box-shadow: 4px 4px 8px rgba(0, 0, 0, 0.3), -4px -4px 8px rgba(255, 255, 255, 0.1);
            border: none;
            cursor: pointer;
            transition: all 0.2s ease-in-out;
            margin-top: 15px;
        }
        .styled-button:active {
            box-shadow: inset 4px 4px 8px rgba(0, 0, 0, 0.3), inset -4px -4px 8px rgba(255, 255, 255, 0.1);
            transform: translateY(2px);
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Centered title
st.markdown("<div class='title'>🔍 Select RAG Application</div>", unsafe_allow_html=True)
st.markdown("<p class='description'>Choose which RAG-based system you want to run</p>", unsafe_allow_html=True)

# UI Layout for selection
cols = st.columns(len(PROGRAMS))

for i, (app_name, (script, port, description, bg_color)) in enumerate(PROGRAMS.items()):
    with cols[i]:
        # Styled App Card with unique background color
        st.markdown(f"""
            <div class='app-card' style='background-color: {bg_color};'>
                {app_name}
                <p class='app-description'>{description}</p>
            </div>
        """, unsafe_allow_html=True)

        # Run button (triggers Python function)
        if st.button(f"▶ Run {app_name}", key=f"run_{port}"):
            start_selected_app(script, port)

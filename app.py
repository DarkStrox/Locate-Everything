import sys
import subprocess
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_script = os.path.join(script_dir, "streamlit_app.py")
    
    print("========================================================")
    print("Starting NVIDIA LocateEverything Streamlit Web UI...")
    print("Connecting to llama_server at http://127.0.0.1:8085")
    print("========================================================")
    
    cmd = [sys.executable, "-m", "streamlit", "run", streamlit_script]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

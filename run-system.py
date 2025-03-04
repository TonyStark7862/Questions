import subprocess
import os
import time
import sys
import threading

def run_streamlit_app():
    print("Starting Streamlit app...")
    # Save streamlit app to a file
    with open("app.py", "w") as f:
        with open("streamlit-social-app.py", "r") as source:
            f.write(source.read())
    
    # Run streamlit app
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return streamlit_process

def run_agent_system():
    print("Starting AI agent system...")
    # Save agent system to a file
    with open("agent_system.py", "w") as f:
        with open("agent-system.py", "r") as source:
            f.write(source.read())
    
    # Run agent system
    agent_process = subprocess.Popen(
        ["python", "agent_system.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return agent_process

def monitor_process(process, name):
    while True:
        output = process.stdout.readline()
        if output:
            print(f"[{name}] {output.strip()}")
        if process.poll() is not None:
            break
    
    error = process.stderr.read()
    if error:
        print(f"[{name}] ERROR: {error}")

def main():
    # Check if required files exist
    for filename in ["streamlit-social-app.py", "agent-system.py"]:
        if not os.path.exists(filename):
            print(f"Error: {filename} not found. Please make sure the files are in the current directory.")
            sys.exit(1)
    
    # Create images directory if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")
        print("Created images directory")
    
    # Start both processes
    streamlit_process = run_streamlit_app()
    time.sleep(2)  # Give streamlit a moment to start
    agent_process = run_agent_system()
    
    # Monitor outputs in separate threads
    streamlit_thread = threading.Thread(target=monitor_process, args=(streamlit_process, "Streamlit"))
    agent_thread = threading.Thread(target=monitor_process, args=(agent_process, "Agent System"))
    
    streamlit_thread.daemon = True
    agent_thread.daemon = True
    
    streamlit_thread.start()
    agent_thread.start()
    
    print("\nBoth systems are running!")
    print("Access the Streamlit app at http://localhost:8501")
    print("Press Ctrl+C to stop both systems\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        streamlit_process.terminate()
        agent_process.terminate()
        print("Done!")

if __name__ == "__main__":
    main()

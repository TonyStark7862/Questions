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
    
    # Make agent system executable
    if os.name != 'nt':  # Not Windows
        os.chmod("agent_system.py", 0o755)
    
    # Run agent system with higher priority
    if os.name == 'nt':  # Windows
        agent_process = subprocess.Popen(
            ["python", "agent_system.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Linux/Mac
        agent_process = subprocess.Popen(
            ["python", "agent_system.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
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
        
    # Make sure CSV files exist with proper headers
    if not os.path.exists("posts.csv"):
        pd.DataFrame(columns=['post_id', 'author', 'content', 'image_path', 'timestamp']).to_csv("posts.csv", index=False)
        print("Created posts.csv")
        
    if not os.path.exists("comments.csv"):
        pd.DataFrame(columns=['comment_id', 'post_id', 'author', 'content', 'parent_comment_id', 'timestamp']).to_csv("comments.csv", index=False)
        print("Created comments.csv")
    
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

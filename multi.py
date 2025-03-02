
import subprocess
import time

def run_script(command):
    """Runs a script using the specified command."""
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    try:
        print("Running combined.py using Miniconda Python...")
        run_script(r'C:/Users/khand/miniconda3/python.exe "C:/Users/khand/OneDrive/Documents/python/ai face and emotion recognition/face/src/combined.py"')

        print("Waiting for 5 seconds...")
        time.sleep(5)

        print("Running main.py using system Python...")
        run_script(r'python "C:/Users/khand/OneDrive/Documents/python/ai face and emotion recognition/main.py"')

        print("Both scripts have finished executing.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}. One of the scripts failed.")

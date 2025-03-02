import pandas as pd

# Define the tasks in a structured format
tasks = [
    ["Planning & Setup", "Define restaurant layout", "High", "Not Started", "", "", ""],
    ["Planning & Setup", "Install Blender & Python dependencies", "High", "Not Started", "", "", ""],
    ["Planning & Setup", "Generate ArUco markers", "Medium", "Not Started", "", "", ""],
    ["3D Environment", "Model restaurant layout in Blender", "High", "Not Started", "", "", ""],
    ["3D Environment", "Set up overhead camera", "High", "Not Started", "", "", ""],
    ["3D Environment", "Place & map ArUco markers", "Medium", "Not Started", "", "", ""],
    ["Robot Simulation", "Model robots in Blender", "High", "Not Started", "", "", ""],
    ["Robot Simulation", "Define physics for robots", "Medium", "Not Started", "", "", ""],
    ["Robot Simulation", "Write Python script for robot movement", "High", "Not Started", "", "", ""],
    ["Robot Simulation", "Implement ArUco marker detection", "High", "Not Started", "", "", ""],
    ["Robot Simulation", "Convert marker positions to movement", "High", "Not Started", "", "", ""],
    ["Interaction", "Animate food pickup & delivery", "Medium", "Not Started", "", "", ""],
    ["Interaction", "Implement multi-robot collision avoidance", "High", "Not Started", "", "", ""],
    ["Interaction", "Build UI for controlling robots", "Medium", "Not Started", "", "", ""],
    ["Optimization", "Optimize marker recognition & speed", "Medium", "Not Started", "", "", ""],
    ["Finalization", "Final testing & improvements", "High", "Not Started", "", "", ""]
]

# Create a DataFrame
columns = ["Phase", "Task", "Priority", "Status", "Start Date", "Deadline", "Notes"]
df = pd.DataFrame(tasks, columns=columns)

# Save to an Excel file
file_name = "Robotic_Restaurant_Progress_Tracker.xlsx"
df.to_excel(file_name, index=False, engine="openpyxl")

print(f"Progress tracker saved as {file_name} 🚀")

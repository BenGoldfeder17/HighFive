import tkinter as tk
import random  # Simulating classification
import threading
import time

# Create the main window
root = tk.Tk()
root.title("Smart Trash Can")
root.geometry("720x1280") 

bin_capacity = 10  # Default: 10 gallons
trash_count = 0
recycle_count = 0
item_volume = 0.15  # Assume each item takes 0.15 gallons

# Function to simulate classifying items continuously (Replace with actual detection)
def classify_item():
    global trash_count, recycle_count
    while True:
        time.sleep(3)  # Simulate scanning every 3 seconds
        item = random.choice(["Recyclable", "Trash"])

        if item == "Recyclable":
            recycle_count += 1
        else:
            trash_count += 1

        root.after(0, update_display, item)

# Function to update the labels on screen
def update_display(item):
    recycle_percentage = min((recycle_count * item_volume / bin_capacity) * 100, 100)
    trash_percentage = min((trash_count * item_volume / bin_capacity) * 100, 100)

    classification_label.config(text=f"Item: {item}")
    recycle_label.config(text=f"Recyclable: {recycle_percentage:.1f}%")
    trash_label.config(text=f"Trash: {trash_percentage:.1f}%")

# Function to reset the counts
def reset_counts():
    global trash_count, recycle_count
    trash_count = 0
    recycle_count = 0
    update_display("Scanning...")

# Function to open the capacity window
def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Set Bin Capacity")
    settings_window.geometry("300x200")

    # Function to save user-selected bin capacity
    def save_capacity():
        global bin_capacity
        bin_capacity = capacity_slider.get()
        settings_window.destroy()

    tk.Label(settings_window, text="Select Bin Size (4-32 gallons):").pack(pady=10)
    capacity_slider = tk.Scale(settings_window, from_=4, to=32, orient="horizontal", length=200)
    capacity_slider.set(bin_capacity)
    capacity_slider.pack()
    
    tk.Button(settings_window, text="Save", command=save_capacity).pack(pady=10)

# Create menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
options_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Options", menu=options_menu)

# Add options to menu
options_menu.add_command(label="Set Bin Capacity", command=open_settings)
options_menu.add_command(label="Reset Counts", command=reset_counts)  

# Labels for classification and fill percentage
classification_label = tk.Label(root, text="Scanning...", font=("Arial", 18), bg="white", width=20, height=2)
classification_label.pack(pady=10)

recycle_label = tk.Label(root, text="Recyclable: 0.0%", font=("Arial", 14), bg="lightblue", width=20)
recycle_label.pack(pady=5)

trash_label = tk.Label(root, text="Trash: 0.0%", font=("Arial", 14), bg="lightcoral", width=20)
trash_label.pack(pady=5)

# Idk wtf is that it just made the code run correctly
threading.Thread(target=classify_item, daemon=True).start()

# Run the GUI
root.mainloop()

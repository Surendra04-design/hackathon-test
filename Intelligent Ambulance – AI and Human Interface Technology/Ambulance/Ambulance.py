from tkinter import messagebox, filedialog, Tk, Label, Button, Text, Scrollbar, END
import tkinter
import numpy as np
import pandas as pd
import socket
import pickle
import time

# Initialize Ambulance GUI window
main = Tk()
main.title("Ambulance Application")
main.geometry("900x500")
main.config(bg='turquoise')

def reportCondition():
    text.delete('1.0', END)
    # Ask user to select a CSV file from the Dataset folder
    filename = filedialog.askopenfilename(initialdir="Dataset", filetypes=[("CSV files", "*.csv")])
    if not filename:
        text.insert(END, "No file selected. Please select a valid CSV file.\n")
        return

    try:
        dataset = pd.read_csv(filename)
    except Exception as e:
        text.insert(END, f"Error reading file: {e}\n")
        return

    dataset.fillna(0, inplace=True)
    dataset = dataset.values
    for i in range(len(dataset)):
        # Convert record to comma-separated string
        data = dataset[i]
        arr = ','.join([str(x) for x in data])
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Connect to the hospital server (ensure PatientMonitoring.py is running)
            client.connect(('localhost', 2222))
        except Exception as e:
            text.insert(END, f"Error connecting to server: {e}\n")
            return

        features = ["patientdata", arr]
        features = pickle.dumps(features)
        client.send(features)
        response = client.recv(1000)
        response = response.decode()
        text.insert(END, f"Patient Test Data = {arr} ===> {response}\n\n")
        client.close()
        text.update_idletasks()
        time.sleep(1)

# GUI Elements
font = ('times', 16, 'bold')
title_label = Label(main, text='Ambulance Application')
title_label.config(bg='dark goldenrod', fg='white', font=font)
title_label.config(height=3, width=120)
title_label.place(x=0, y=5)

font1 = ('times', 13, 'bold')
report_btn = Button(main, text="Report Patient Condition to Hospital Server", command=reportCondition)
report_btn.place(x=200, y=450)
report_btn.config(font=font1)

font2 = ('times', 12, 'bold')
text = Text(main, height=18, width=110, font=font2)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)

main.mainloop()

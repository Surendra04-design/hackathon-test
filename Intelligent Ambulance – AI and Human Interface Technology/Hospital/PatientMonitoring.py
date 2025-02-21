from tkinter import messagebox, filedialog, Tk, Label, Button, Text, Scrollbar, END
import tkinter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import socket
from threading import Thread
import pickle

# Initialize PatientMonitoring (Hospital Server) GUI window
main = Tk()
main.title("Intelligent Ambulance – AI and Human Interface Technology")
main.geometry("1300x1200")
main.config(bg='turquoise')

# Global variables for dataset, model, and scaling
global filename, dataset, X_train, X_test, y_train, y_test, classifier, X, Y, sc
accuracy = []
precision = []
recall = []
fscore = []
labels = ['Normal Heart Rate', 'Abnormal Heart Rate']

def upload():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset", filetypes=[("CSV files", "*.csv")])
    if not filename:
        text.insert(END, "No file selected. Please select a valid CSV file.\n")
        return
    text.insert(END, f"{filename} dataset loaded\n")
    try:
        dataset = pd.read_csv(filename)
    except Exception as e:
        text.insert(END, f"Error reading file: {e}\n")
        return
    dataset.fillna(0, inplace=True)
    text.insert(END, str(dataset.head())+"\n")
    text.insert(END, "Dataset contains total records    : " + str(dataset.shape[0]) + "\n")
    text.insert(END, "Dataset contains total attributes : " + str(dataset.shape[1]) + "\n")
    label_counts = dataset.groupby('target').size()
    label_counts.plot(kind="bar")
    plt.show()

def processDataset():
    global X_train, X_test, y_train, y_test, sc, X, Y, dataset
    text.delete('1.0', END)
    data = dataset.values
    X = data[:, 0:data.shape[1]-1]
    Y = data[:, data.shape[1]-1].astype(int)
    # Shuffle the dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    text.insert(END, "Dataset Preprocessing, Normalizing & Shuffling Task Completed\n")
    text.insert(END, str(X) + "\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END, "Dataset train and test split details\n\n")
    text.insert(END, "80% records used for Ensemble training Algorithm : " + str(X_train.shape[0]) + "\n")
    text.insert(END, "20% records used for Ensemble testing Algorithm  : " + str(X_test.shape[0]) + "\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END, f"{algorithm} Accuracy  : {a}\n")
    text.insert(END, f"{algorithm} Precision : {p}\n")
    text.insert(END, f"{algorithm} Recall    : {r}\n")
    text.insert(END, f"{algorithm} FSCORE    : {f}\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title(f"{algorithm} Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()

def runDecisionTree():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, X_train, X_test, y_train, y_test
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    dt_cls = DecisionTreeClassifier() 
    dt_cls.fit(X_train, y_train)
    predictions = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree", y_test, predictions)

def runRandomForest():
    global X_train, X_test, y_train, y_test, classifier
    rf_cls = RandomForestClassifier() 
    rf_cls.fit(X_train, y_train)
    classifier = rf_cls  # Save the trained model for prediction requests
    predictions = rf_cls.predict(X_test)
    calculateMetrics("Random Forest", y_test, predictions)

def runKNN():
    global X_train, X_test, y_train, y_test, classifier
    knn_cls = KNeighborsClassifier(n_neighbors=10) 
    knn_cls.fit(X_train, y_train)
    predictions = knn_cls.predict(X_test)
    calculateMetrics("KNN", y_test, predictions)

def graph():
    df = pd.DataFrame([
        ['Decision Tree', 'Precision', precision[0]],
        ['Decision Tree', 'Recall', recall[0]],
        ['Decision Tree', 'F1 Score', fscore[0]],
        ['Decision Tree', 'Accuracy', accuracy[0]],
        ['Random Forest', 'Precision', precision[1]],
        ['Random Forest', 'Recall', recall[1]],
        ['Random Forest', 'F1 Score', fscore[1]],
        ['Random Forest', 'Accuracy', accuracy[1]],
        ['KNN', 'Precision', precision[2]],
        ['KNN', 'Recall', recall[2]],
        ['KNN', 'F1 Score', fscore[2]],
        ['KNN', 'Accuracy', accuracy[2]],
    ], columns=['Parameters', 'Algorithms', 'Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

running = True

def startCloudServer():
    global text
    # Thread class to handle each incoming connection
    class CloudThread(Thread):
        def __init__(self, conn, ip, port):
            Thread.__init__(self)
            self.conn = conn
            self.ip = ip
            self.port = port
            print(f"Request received from Ambulance IP: {ip} with port no: {port}")
 
        def run(self):
            data = self.conn.recv(1000)
            try:
                received = pickle.loads(data)
            except Exception as e:
                text.insert(END, f"Error decoding data: {e}\n")
                self.conn.close()
                return
            request = received[0]
            if request == "patientdata":
                patient_data = received[1]
                text.delete('1.0', END)
                text.insert(END, f"Patient Data Received: {patient_data}\n")
                text.update_idletasks()
                output = predict(patient_data)
                self.conn.send(output.encode())
                text.insert(END, output + "\n\n")
            self.conn.close()
            
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('localhost', 2222))
    threads = []
    text.insert(END, "Cloud Server Started\n\n")
    while running:
        server.listen(4)
        conn, (ip, port) = server.accept()
        new_thread = CloudThread(conn, ip, port)
        new_thread.start()
        threads.append(new_thread)
    for t in threads:
        t.join()

def startServer():
    Thread(target=startCloudServer).start()

def predict(data):
    try:
        values = [float(x) for x in data.split(",")]
    except Exception as e:
        return f"Error converting data: {e}"
    return predictCondition(values)

def predictCondition(testData):
    global sc, classifier
    arr = np.array([testData])
    arr = sc.transform(arr)
    try:
        prediction = classifier.predict(arr)
    except Exception as e:
        return f"Prediction error: {e}"
    pred = prediction[0]
    msg = "Predicted Output: Patient Condition Normal"
    if pred == 1:
        msg = "Predicted Output: Patient Condition Abnormal"
    return msg

# GUI Elements for PatientMonitoring Application
font = ('times', 16, 'bold')
title_label = Label(main, text='Intelligent Ambulance – AI and Human Interface Technology')
title_label.config(bg='dark goldenrod', fg='white', font=font)
title_label.config(height=3, width=120)
title_label.place(x=0, y=5)

font1 = ('times', 13, 'bold')
upload_btn = Button(main, text="Upload Heart Disease Dataset", command=upload)
upload_btn.place(x=890, y=100)
upload_btn.config(font=font1)

process_btn = Button(main, text="Dataset Preprocessing & Train Test Split", command=processDataset)
process_btn.place(x=890, y=150)
process_btn.config(font=font1)

dt_btn = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dt_btn.place(x=890, y=200)
dt_btn.config(font=font1)

rf_btn = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rf_btn.place(x=890, y=250)
rf_btn.config(font=font1)

knn_btn = Button(main, text="Run KNN Algorithm", command=runKNN)
knn_btn.place(x=890, y=300)
knn_btn.config(font=font1)

graph_btn = Button(main, text="Comparison Graph", command=graph)
graph_btn.place(x=890, y=350)
graph_btn.config(font=font1)

predict_btn = Button(main, text="Receive Patient Condition to Hospital Server", command=startServer)
predict_btn.place(x=890, y=400)
predict_btn.config(font=font1)

text = Text(main, height=30, width=100, font=('times', 12, 'bold'))
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)

main.mainloop()

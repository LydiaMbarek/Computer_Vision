from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog
import numpy as np
import subprocess

import Functions


h = 500
w = 500
path = None


def to_pil(img, label, index,color = 0):
    global w, h
    if color == 0:
        width,high,_ = img.shape
    else:
        width,high = img.shape
    if w < width or h < high:
        img = cv2.resize(img, (w, h))
    img = cv2.flip(img, 1)

    if color ==0:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            pass
        image = Image.fromarray(img)
        pic = ImageTk.PhotoImage(image)
        label.configure(image=pic)
        label.image = pic
        label.grid(row=10, column=index, padx=10, pady=10)
    else:
        image = Image.fromarray(img)
        pic = ImageTk.PhotoImage(image)
        label.configure(image=pic)
        label.image = pic
        label.grid(row=10, column=index, padx=10, pady=10)

def show():
    global frame,path
    if com3.get() == "Camera":
        _, frame = cap.read()
    if com3.get() == "Image":
        frame = image
    to_pil(frame, label,0)
    if switch.get() == "Green Screen":
        if path == None:
            path = filedialog.askopenfilename(title="Select Image BackGround",filetypes=(('JPG', '*.jpg'), ('PNG', '*.png')))
        imagebackGroud = cv2.imread(path)
        imagebackGroud = cv2.resize(imagebackGroud, (w, h))
        frame = cv2.resize(frame, (w, h))
        result = Functions.GreenScreen(frame, imagebackGroud, low, high)
        to_pil(result, label_2,1)
    elif switch.get() == "Invisible cloak":
        result = Functions.invisibility_cloak(frame,low, high)
        to_pil(result, label_2,1)
    elif switch.get() == "Detect Object":
        result = Functions.detect_Image(frame,low,high)
        to_pil(result, label_2, 1)
    elif switch.get() == "Moyenne":
        result = Functions.blur_3d(frame,size_kernel=kernel_size)
        to_pil(result, label_2, 1)
    elif switch.get() == "Mode":
        input = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        result = Functions.filtreMedian(input,kernel_size)
        to_pil(result, label_2,1,2)
    elif switch.get() == "Gausse":
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = Functions.apply_gaussian_filter(input, kernel_size)
        to_pil(result, label_2, 1, 2)
    elif switch.get() == "Laplacian":
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = Functions.apply_laplacian_filter(input)
        to_pil(result, label_2, 1, 2)
    elif switch.get() == "Dilation":
        img = Functions.inRange(frame, low, high)
        kernal = np.ones((kernel_size,kernel_size))
        img = Functions.dilate(img, kernal)
        to_pil(img, label_2, 1, 2)
    elif switch.get() == "Erasion":
        img = Functions.inRange(frame, low, high)
        kernal = np.ones((kernel_size, kernel_size))
        img = Functions.erode(img, kernal)
        to_pil(img, label_2, 1, 2)
    elif switch.get() == "Morphologye OPEN":
        img = Functions.inRange(frame,low,high)
        img = Functions.morphEx(img,kernel_size,Functions.OPEN)
        to_pil(img, label_2, 1, 2)
    elif switch.get() ==  "Morphologye CLOSE":
        img = Functions.inRange(frame, low, high)
        img = Functions.morphEx(img, kernel_size, Functions.CLOSE)
        to_pil(img, label_2, 1, 2)
    elif switch.get() == "Kernal":
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_image = Functions.apply_kernel(input)
        img = np.uint8(filtered_image)  # Convert image data to uint8 if needed
        to_pil(img, label_2, 1, 2)
    elif switch.get() == "Bilateral":
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        filtered_image = Functions.bilateral(input,  kernel_size=(kernel_size, kernel_size), sigma_s=1.5, sigma_r=0.1)
        img = np.uint8(filtered_image)  # Convert image data to uint8 if needed
        to_pil(img, label_2, 1, 2)
    if com3.get() == "Camera":
        label.after(30, show)
def choose():
    global cap, image, low, high,kernel_size
    if com3.get() == "Camera":
        if cap == None:
            cap = cv2.VideoCapture(0)
        else:
            label.after_cancel(show)
    if com3.get() == "Image":
        path = filedialog.askopenfilename(title="Select file", filetypes=(('JPG', '*.jpg'), ('PNG', '*.png')))
        image = cv2.imread(path)
    try:
        kernel_size = int(kernel_size_entry.get())
        if kernel_size % 2 != 0:
            low = np.array([int(low_h.get()), int(low_s.get()), int(low_v.get())])
            high = np.array([int(high_h.get()), int(high_s.get()), int(high_v.get())])
            show()
        else:
            messagebox.showwarning("Warning", "size kernal must be odd integer")

    except:
        messagebox.showwarning("Warning", "size kernal must be odd integer")

def play_game():
    subprocess.Popen(["python", "game_display1.py"])

win = Tk()
win.geometry("800x800")
cap = None
items = "Camera", "Image"
com3 = ttk.Combobox(win, values=items)
play_game_button = Button(win, text='Play Game', command=play_game)
com3.current(0)
label_3 = Label(win, text="Source Image")
label = Label(win)
label_2 = Label(win)
button3 = Button(win, text='Appliquer', command=choose)
label_4 = Label(win, text="Opération")
imgs = "Detect Object", "Green Screen", "Invisible cloak", "Moyenne", "Mode", "Gausse", "Laplacian","Dilation","Erasion", "Morphologye OPEN","Morphologye CLOSE","Kernal","Bilateral"
switch = ttk.Combobox(win, values=imgs)
switch.current(0)
kernel_size_label = Label(win, text="Kernel Size:")
kernel_size_entry = Entry(win, validate="key", validatecommand=(win.register(lambda s: s.isdigit() and int(s) % 2 != 0), '%P'))
low_hsv_label = Label(win, text="Low HSV")
high_hsv_label = Label(win, text="High HSV")

# Create Spinboxes for Low and High HSV values
low_h = Spinbox(win, from_=0, to=255, width=5)
low_h.delete(0, "end")  # Clear the default entry
low_h.insert(0, str(95))
low_s = Spinbox(win, from_=0, to=255, width=5)
low_s.delete(0, "end")  # Clear the default entry
low_s.insert(0, str(80))
low_v = Spinbox(win, from_=0, to=255, width=5)
low_v.delete(0, "end")  # Clear the default entry
low_v.insert(0, str(60))

high_h = Spinbox(win, from_=0, to=255, width=5)
high_h.delete(0, "end")
high_h.insert(0, str(115))
high_s = Spinbox(win, from_=0, to=255, width=5)
high_s.delete(0, "end")
high_s.insert(0, str(255))
high_v = Spinbox(win, from_=0, to=255, width=5)
high_v.delete(0, "end")
high_v.insert(0, str(150))

# Place the labels and Spinboxes on the grid
low_hsv_label.grid(row=0, column=4, padx=10, pady=10)
low_h.grid(row=1, column=4, padx=10, pady=10)
low_s.grid(row=2, column=4, padx=10, pady=10)
low_v.grid(row=3, column=4, padx=10, pady=10)

high_hsv_label.grid(row=0, column=5, padx=10, pady=10)
high_h.grid(row=1, column=5, padx=10, pady=10)
high_s.grid(row=2, column=5, padx=10, pady=10)
high_v.grid(row=3, column=5, padx=10, pady=10)


com3.grid(row=0, column=1, padx=10, pady=10)
button3.grid(row=0, column=10, padx=10, pady=10)
label_3.grid(row=0, column=0, padx=10, pady=10)
label_4.grid(row = 1, column=0)
switch.grid(row=1, column=1, padx=10, pady=10)
play_game_button.grid(row=1, column=10, padx=10, pady=10)
kernel_size_label.grid(row=2, column=0, padx=10, pady=10)
kernel_size_entry.grid(row=2, column=1, padx=10, pady=10)

win.mainloop()

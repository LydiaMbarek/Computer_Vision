import Functions
import time
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import random
import keyboard
gameover = 0

# les valeur ta3 score marahomch ytbedlo ki n3ayetelhom f create_gray_side() !! malgri ram global  ???
score = 0
high_score = 0
speedGame = 0.3
label11 = None
message_label = None
label22 = None
label33 = None
high_board = None
game_over_shown = False
colorOurCar = None
colorBackGroud = None
listCars = None
x_Left = None
heighWith = None
car_width = None
imagePrincipal = None

def createBackGround(widh_game,high_game, colorBackGround):
    return np.ones((high_game, widh_game, 3), dtype=np.uint8) * 195
def createCar(image,posY, heigh_car, width_car, colorCar):
    height, weight, _ = image.shape
    for i in range(posY,posY+width_car):
        for j in range(0,heigh_car):
            image[j, i] = colorCar
    return image
def creatRect(image,posY,posX, posYM, posXM, colorCar):
    height, weight, _ = image.shape
    for i in range(posY,posYM):
        for j in range(posX,posXM):
            image[i, j] = colorCar
    return image
def MovePrincipalCar(image,newPosX ,posY,posX, high_car,width_car,colorCar,colorBackgrond):
    if newPosX != posX:
        posY = posY
        posX = posX
        posYM = posY + high_car
        posXM = posX + width_car
        image = creatRect(image, posY, posX, posYM, posXM, colorBackgrond)
        posY = posY
        posX = newPosX
        posYM = posY + high_car
        posXM = posX + width_car
        image = creatRect(image, posY, posX, posYM, posXM, colorCar)
    return image
def create_ourCare(image,posy ,posX, heigh_car, width_car, colorCar):
    height, weight, _ = image.shape
    for i in range(posX,posX+width_car):
        for j in range(posy,posy + heigh_car):
            image[j,i] = colorCar
    return image
def moveCar(image, Newposx, pasPosX, width, heigh, colorBackround, colorCar, carY, thickness):
    if Newposx != pasPosX:
        cv2.rectangle(image,(pasPosX,carY),(pasPosX+width,carY),colorBackround,thickness)
        cv2.rectangle(image,(Newposx,carY),(Newposx+width,carY),colorCar,thickness)
    return image
def moveCars(image, oldPoints ,width, high, pas,colorBackGround,colorCars):
    global score , speedGame , game_over_shown
    i = 0
    while i < len(oldPoints):
        posY = oldPoints[i][0]
        posX = oldPoints[i][1]
        posYM = posY + high
        posXM = posX + width
        image = creatRect(image,posY,posX,posYM,posXM,colorBackGround)
        newPoints = [oldPoints[i][0]+pas, oldPoints[i][1]]
        if (newPoints[0] + high > image.shape[0]):
            oldPoints.pop(i)
            i = i - 1
            if not game_over_shown :
                score = score + 1
                label22.config(text=str(score))
                speedGame += 0.1 if score % 100 == 0 else 0
                label33.config(text=str(speedGame))
        else:
            posY = newPoints[0]
            posX = newPoints[1]
            posYM = posY + high
            posXM = posX + width
            image = creatRect(image, posY, posX, posYM, posXM, colorCars)
            oldPoints[i] = newPoints
        i = i + 1
    return image, oldPoints
def checkGame(pointCars,pointCarPrincipale, widthCar,HeighCar):
    for point in pointCars:
        if point[0] <= pointCarPrincipale[0] <= point[0] + HeighCar and pointCarPrincipale[1] == point[1]:
            return True
    return False
# Define the function to create the game window and canvas
def game_over(parent):
    global  message_label
    # Create a label with the message
    message_label = tk.Label(parent, text="Game Over!", font=("Arial", 18),fg="red")
    message_label.place(relx=0.5, rely=0.5, anchor="center")

def reset_game():
    global x_right, listCars , high_score , score, speed, running, game_over_shown , label11 , label22
    x_right = width_board //2
    score = 0
    speed = 1
    running = True
    game_over_shown = False
    label11.config(text=str(high_score))
    label22.config(text=str(score))


def create_white_side(parent):
    white_frame = tk.Frame(parent, bg="white", width= 350, height=600)
    white_frame.pack(side=tk.LEFT)
    return white_frame
def restart():
    print('Je suis la ')
    global  game_over_shown , score , speedGame, label22 , label33 , listCars , high_board , heighWith , x_Left
    global  colorBackGroud , colorOurCar , colorCar , ourCar , imagePrincipal , message_label , gameover
    if game_over_shown : deleateGameOver()
    gameover = 0
    game_over_shown = False
    score = 0
    speedGame = 0.3
    label22.config(text=str(score))
    label33.config(text=str(speedGame))

    ourCar = [high_board - heighWith, x_Left]
    imagePrincipal = createBackGround(width_board, high_board, colorBackGroud)
    listCars = []
    ourCar = [high_board - heighWith, x_Left]
    imagePrincipal = create_ourCare(imagePrincipal, ourCar[0], ourCar[1], heighWith, car_width, colorOurCar)

def create_gray_side(parent):
    global score,high_score,speedGame , label11 , label22 , label33
    score = 0
    high_score = 0
    speedGame = 0.3
    gray_frame = tk.Frame(parent, width=200, height=600)
    gray_frame.pack()
    # Create the first label and text field
    label1 = tk.Label(gray_frame,  text=" High Score ", fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label1.pack(pady=10)
    label11 = tk.Label(gray_frame,  text=str(high_score), fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label11.pack(pady=10)

    # Create the first label and text field
    label2 = tk.Label(gray_frame,  text=" Score ", fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label2.pack()

    label22 = tk.Label(gray_frame,  text=str(score), fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label22.pack()

    # Create the first label and text field
    label3 = tk.Label(gray_frame, text=" Speed ", fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label3.pack()
    label33 = tk.Label(gray_frame, text=str(speedGame), fg="black",font=("Arial", 14), padx=0, pady=0, highlightthickness=0, highlightbackground=gray_frame.cget('bg'))
    label33.pack()

    # Create a button to restart the application
    restart_button = tk.Button(gray_frame, text="Restart", command=restart, bg="red", fg="white",font=("Arial", 14))
    restart_button.pack()

def create_game_window():
    root = tk.Tk()
    root.title("Game Window")
    rootgame = create_white_side(root)
    create_gray_side(root)
    canvas = tk.Canvas(rootgame, width=300, height=600)
    canvas.pack()
    return root,rootgame, canvas

def deleateGameOver():
    global message_label
    message_label.destroy()
def LuncheGame(rootgame, canvas):
    global score,high_score,speedGame , label33 , game_over_shown , x_Left , heighWith , car_width , imagePrincipal
    global img_tk , listCars, high_board , colorCar , colorOurCar , message_label , gameover
    gameover = 0
    score = 0
    speedGame = 0.3
    hard_max = 10
    global width_board
    width_board = 300
    high_board = 600
    x_Left  = 0
    x_right = width_board//2
    colorBackGroud = (195,195,195)
    colorCars = (255,255,255)
    colorOurCar = (255,0,0)
    car_width = 150
    heighWith = 50
    imagePrincipal = createBackGround(width_board,high_board,colorBackGroud)
    listCars = []
    ourCar = [high_board-heighWith,x_Left]
    pas = 20
    start_time = time.time()
    eachCarOut = time.time()
    low = np.array([95, 80, 60])
    high = np.array([115, 255, 150])
    game_over(rootgame)
    deleateGameOver()
    imagePrincipal = create_ourCare(imagePrincipal, ourCar[0], ourCar[1], heighWith, car_width, colorOurCar)
    videoCape = cv2.VideoCapture(0)
    while (True):
        global  label11 , label22
        ret, frame = videoCape.read()
        cv2.flip(frame, 1, frame)
        weight_frame = frame.shape[1]
        center =  Functions.detect_Center(frame,low,high)
        center_frame = weight_frame // 2
        if keyboard.is_pressed('left'):
            imagePrincipal = MovePrincipalCar(imagePrincipal, x_Left, ourCar[0], ourCar[1], heighWith, car_width, colorOurCar, colorBackGroud)
            ourCar = [high_board - heighWith, x_Left]
        elif keyboard.is_pressed('right'):
            imagePrincipal = MovePrincipalCar(imagePrincipal, x_right, ourCar[0], ourCar[1], heighWith, car_width, colorOurCar, colorBackGroud)
            ourCar = [high_board - heighWith, x_right]
        if center is not None:
            if center_frame  < center[1] :
                imagePrincipal = MovePrincipalCar(imagePrincipal, x_right, ourCar[0], ourCar[1], heighWith, car_width,
                                              colorOurCar, colorBackGroud)
                ourCar = [high_board - heighWith, x_right]
            elif center_frame > center[1] :
                imagePrincipal = MovePrincipalCar(imagePrincipal, x_Left, ourCar[0], ourCar[1], heighWith, car_width,
                                              colorOurCar, colorBackGroud)
                ourCar = [high_board - heighWith, x_Left]
        if (time.time() - eachCarOut) >= hard_max:
            x = random.randrange(2)
            if x == 0:
                listCars.append([0,x_Left])
                imagePrincipal = createCar(imagePrincipal, x_Left, heighWith, car_width,colorCars)
            else:
                listCars.append([0,x_right])
                imagePrincipal = createCar(imagePrincipal, x_right, heighWith, car_width,colorCars)
            eachCarOut = time.time()
        if (time.time() - start_time) >= speedGame:
            imagePrincipal, listCars = moveCars(imagePrincipal, listCars, car_width, heighWith,pas, colorBackGroud,colorCars)
            start_time = time.time()
        if checkGame(listCars,ourCar,car_width,heighWith):
            if score > high_score:
                high_score = score
                label11.config(text=str(high_score))
            game_over_shown = True
            if gameover == 0 :
                game_over(rootgame)
                gameover = 1

        #speed += 0.1 if score % 100 == 0 else 0


        img = Image.fromarray(imagePrincipal)
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        rootgame.update()  # Update the window in the main loop
    # Close the window when the loop ends
    rootgame.destroy()


def lancerWindowGame():
    root, rootgame, canvas = create_game_window()
    LuncheGame(rootgame, canvas)

lancerWindowGame()
# Start the game




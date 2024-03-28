import time

import Functions
from  Functions import trouver_contours_center

import cv2
import numpy as np
import random
import keyboard


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
def LuncheGame():
    hard_max = 10
    speedGame = 0.3
    width_board = 300
    high_board = 820
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

    imagePrincipal = create_ourCare(imagePrincipal, ourCar[0], ourCar[1], heighWith, car_width, colorOurCar)
    videoCape = cv2.VideoCapture(0)
    while (True):
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
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cv2.imshow("Game Show",imagePrincipal)
    cv2.destroyAllWindows()
LuncheGame()


import cv2
import numpy as np
from centroidTracker import CentroidTracker

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# retorna uma mascara de cores definida pelos parametros lower e upper
def findColor(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    return mask

# Pega os contornos do jogadores e retorna o ponto x e y dele no video
def getContours(img, image, cor, pontos, trackers):
     #Remove ruidos com a gaussiana.
    imagem_borrada = cv2.GaussianBlur(img, (5, 5), 0)
    
    #aplica um limiar (trasehold). Está usando um núcleo invertido para ser utilizado no escuro para remover o fundo com facilidade
    _, imagem_borrada_e_limiarizada = cv2.threshold(imagem_borrada, 70, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contours,hierarchy = cv2.findContours(imagem_borrada_e_limiarizada,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for cnt in contours:
        #print(cnt)
        area = cv2.contourArea(cnt)
        if area<1000:
            #cv2.drawContours(image, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            ponto_jogador = (int(x+w/2),int(y+h/2))
            pontos.append([ponto_jogador, cor])
            if(cor[0] == 255):
                rect = (x, y, w, h)
                
                tracker = cv2.TrackerKCF_create()
                trackers.add(tracker, image, rect)

#desenha um circulo no ponto x, y passado
def desenhaCaminho(image, pontos):
    for point in pontos:
        cv2.circle(image, point[0], 10, point[1], cv2.FILLED)

def main():
    trackers = cv2.MultiTracker_create()

    #Abre o vídeo gravado em disco
    camera = cv2.VideoCapture('run.mp4')

    (sucesso, frame) = camera.read()

    #todos os pontos desenhados
    pontos = []

    while True:
        #read() retorna 1-Se houve sucesso e 2-O próprio frame
        (sucesso, frame) = camera.read()
        if not sucesso: #final do vídeo
            #recomeca o video
            camera = cv2.VideoCapture('run.mp4')
            (sucesso, frame) = camera.read()

        (success, boxes) = trackers.update(frame)

        if success:
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        
        image = frame.copy()
        novosPontos = []

        # arrays com lowers e upper das cores azul e vermelho do video
        lower_blue = np.array([96,188,175])
        upper_blue = np.array([155,255,255])

        lower_red = np.array([159,0,0])
        upper_red = np.array([255,255,255])

        #mascara encontrada usando a funcao inRange
        #mask_red = findColor(image, lower_red, upper_red)
        mask_blue = findColor(image, lower_blue, upper_blue)
        
        #desenha contornos na imagem
        getContours(mask_blue, image, (255,0,0), novosPontos, trackers)
        #getContours(mask_red, image, (0,0,255), novosPontos, boxes)
           

        if(len(novosPontos) != 0):
            for p in novosPontos:
                pontos.append(p)
        
        # if(len(pontos) != 0):
        #     desenhaCaminho(image, pontos)

        #resultado do video ainda vai alterar
        res_blue = cv2.bitwise_and(image,image, mask= mask_blue)                                                                                                                                           
        #res_red = cv2.bitwise_and(image,image, mask= mask_blue)                

        img_stacked = stackImages(0.4, ([frame,mask_blue], [image,res_blue]))
        cv2.imshow("Exibindo video: Time vermelho | Time azul", img_stacked)

        key = cv2.waitKey(1) & 0xFF

        #Espera que a tecla 'q' seja pressionada para sair
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
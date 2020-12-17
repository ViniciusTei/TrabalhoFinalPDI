import cv2
import numpy as np
from centroidTracker import CentroidTracker
from collections import OrderedDict
from scipy.spatial import distance
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
def getContours(img, image, cor):
     #Remove ruidos com a gaussiana.
    imagem_borrada = cv2.GaussianBlur(img, (5, 5), 0)
    
    #aplica um limiar (trasehold). Está usando um núcleo invertido para ser utilizado no escuro para remover o fundo com facilidade
    _, imagem_borrada_e_limiarizada = cv2.threshold(imagem_borrada, 70, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    contours,hierarchy = cv2.findContours(imagem_borrada_e_limiarizada,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    rects = []
    for cnt in contours:
        #print(cnt)
        area = cv2.contourArea(cnt)
        #print(area)
        if area<450:
            #cv2.drawContours(image, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            x, y, w, h = cv2.boundingRect(approx)
            rects.append((x, y, x+w, y+h))

    return rects

def calculaDistancia(initialDict, currentDict):
    dist = OrderedDict()

    for key, value in initialDict.items():
        for current_key, current_value in currentDict.items():
            if key == current_key:
                dist[key] = value[0] - current_value[0]

    return dist

def defineAtaqueOuDefesa(distancias):
    values = distancias.values()
    soma = 0
    for value in values:
        soma += value
    
    if soma > 0:
        result = 'ATACANDO'
    else:
        result = 'DEFENDENDO'
    
    return result

    
def main():
    ct = CentroidTracker()
    #Abre o vídeo gravado em disco
    camera = cv2.VideoCapture('run.mp4')
    first_object = OrderedDict()
    distancias = OrderedDict()
    firstFrame = True
    count_frames = 0
    result = 'Nothing'
    while True:
        #read() retorna 1-Se houve sucesso e 2-O próprio frame
        (sucesso, frame) = camera.read()
        
        count_frames += 1
        
        if not sucesso: #final do vídeo
            #recomeca o video
            camera = cv2.VideoCapture('run.mp4')
            (sucesso, frame) = camera.read()
        
        image = frame.copy()

        # arrays com lowers e upper das cores azul e vermelho do video
        lower_blue = np.array([70,150,150])
        upper_blue = np.array([155,255,255])

        lower_red = np.array([159,0,0])
        upper_red = np.array([255,255,255])

        #mascara encontrada usando a funcao inRange
        #mask_red = findColor(image, lower_red, upper_red)
        mask_blue = findColor(image, lower_blue, upper_blue)
        
        #desenha contornos na imagem
        bboxes = getContours(mask_blue, image, (0,0,255))

        new_objects = ct.update(bboxes)
        #loop over the tracked objects
        if firstFrame:
            first_object = new_objects.copy()
            firstFrame = False
        else:
            distancias = calculaDistancia(first_object, new_objects)
            if count_frames == 30:
                #analiza ataque ou defesa
                result = defineAtaqueOuDefesa(distancias)
                count_frames = 0

        #escreve na tela o resultado
        cv2.putText(image, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for (objectID, centroid) in new_objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame            
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

        #resultado do video ainda vai alterar
        res_blue = cv2.bitwise_and(image,image, mask= mask_blue)                                                                                                                                           
        #res_red = cv2.bitwise_and(image,image, mask= mask_blue)                

        img_stacked = stackImages(0.4, ([frame,mask_blue], [image,res_blue]))
        #cv2.imshow("Exibindo video: Time azul", img_stacked)
        
        cv2.imshow("Result", image)
        key = cv2.waitKey(1) & 0xFF

        #Espera que a tecla 'q' seja pressionada para sair
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
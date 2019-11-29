import cv2
import numpy as np
import operator
import os

from keras.models import load_model


image = cv2.imread('C:/photo/park.png')

lower = np.uint8([120, 120, 120])
upper = np.uint8([255, 255, 255])

white_mask = cv2.inRange(image, lower, upper)

img = cv2.bitwise_and(image, image, mask = white_mask)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 200)
rows, cols = image.shape[:2]
pt_1 = [cols*0.05, rows*0.90]
pt_2 = [cols*0.05, rows*0.70]
pt_3 = [cols*0.30, rows*0.55]
pt_4 = [cols*0.60, rows*0.15]
pt_5 = [cols*0.90, rows*0.15]
pt_6 = [cols*0.90, rows*0.95]

vertices = np.array([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6], dtype=np.int32)
point_img = canny.copy()
point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2BGR)

for point in vertices:
	cv2.circle(point_img, (point[0], point[1]), 10, (0, 0, 255), 4)

mask = np.zeros_like(gray)
cv2.fillPoly(mask, [vertices], 255)
bitand = cv2.bitwise_and(canny, mask)

lines = cv2.HoughLinesP(bitand, rho=0.1, theta=np.pi/10, threshold=15, minLineLength=9, maxLineGap=4)
select = np.copy(image)
clearned = []
for line in lines:
	for x1, y1, x2, y2 in line:
		if abs(y2-y1) <=1 and abs(x2-x1) >=15 and abs(x2-x1) <=55:
			clearned.append((x1, y1, x2, y2))
			cv2.line(select, (x1, y1), (x2, y2), [200, 0, 0], 2)

list1 = sorted(clearned, key=operator.itemgetter(0, 1))


clusters = {}
dIndex = 0
clus_dist = 10
for i in range(len(list1)-1):
	distance = abs(list1[i+1][0]-list1[i][0])
	if distance <= clus_dist:
		if not dIndex in clusters.keys(): clusters[dIndex] = []

		clusters[dIndex].append(list1[i])
		clusters[dIndex].append(list1[i+1])

	else:
		dIndex+= 1
#print(clusters[0])


rects = {}
i = 0

for key in clusters:
    all_list=clusters[key]
    cleaned=list(set(all_list))
    #print(cleaned)

    
    if len(cleaned)>5:
        cleaned=sorted(cleaned,key=lambda x:x[1])
        avg_y1=cleaned[0][1]
        avg_y2=cleaned[-1][1]
        avg_x1=0
        avg_x2=0
        for tup in cleaned:
            avg_x1 += tup[0]
            avg_x2 += tup[2]

        avg_x1=avg_x1/len(cleaned)
        avg_x2=avg_x2/len(cleaned)
        rects[i]=(avg_x1,avg_y1,avg_x2,avg_y2)
        i += 1
#print(rects)


new_image = image.copy()
buff = 7
for key in rects:
	tup_topLeft = (int(rects[key][0]-buff), int(rects[key][1]))
	tup_botRight = (int(rects[key][2]+buff), int(rects[key][3]))
	cv2.rectangle(new_image, tup_topLeft, tup_botRight, (0, 255, 0), 1)




kp_image = image.copy()
gap = 9
spot_dict = {}
tot_spots = 0
adj_y1 = {0:-5, 1:-10, 2:-10, 3:-11, 4:-15, 5:-5, 6:-15, 7:-15, 8:-15, 9:-10, 10:9, 11:-5}
adj_y2 = {0:10, 1:10, 2:15, 3:10, 4:-5, 5:5, 6:-5, 7:0, 8:25, 9:20, 10:0, 11:30}

adj_x1 = {0:0, 1:-5, 2:-10, 3:-10, 4:-7, 5:-7, 6:-7, 7:-7, 8:-5, 9:-5, 10:-5, 11:0}
adj_x2 = {0:5, 1:10, 2:10, 3:10, 4:7, 5:7, 6:7, 7:7, 8:5, 9:5, 10:5, 11:0}

for key in rects:
	tup = rects[key]
	x1 = int(tup[0]+adj_x1[key])
	x2 = int(tup[2]+adj_x2[key])
	y1 = int(tup[1]+adj_y1[key])
	y2 = int(tup[3]+adj_y2[key])
	cv2.rectangle(kp_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
# cv2.imshow('kp_image', kp_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#    //  舍弃小数部分

	num_splits=int(abs(y2-y1)/gap)
	for i in range(0,num_splits+1):
	    y=int(y1+i*gap)
	    cv2.line(kp_image,(x1,y),(x2,y),(0,0,255),1)
	    cv2.imshow('kp_image', kp_image)




	    if key>0 and key<len(rects)-1:
	        #竖直线
	        x=int((x1+x2)/2)
	        cv2.line(kp_image,(x,y1),(x,y2),(255,0,0),1)

	    #计算数量
	    if key ==0 or key ==(len(rects)-1):
	        tot_spots += num_splits+1
	    else:
	        tot_spots += 2*(num_splits+1)
	    
	    #字典对应好
	    if key ==0 or key ==(len(rects)-1):
	        for i in range(0,num_splits+1):
	            cur_len=len(spot_dict)

	            y=int(y1+i*gap)
	            spot_dict[(x1,y,x2,y+gap)]=cur_len+1
	    
	    else:

	    	for i in range(0,num_splits):
	            cur_len=len(spot_dict)
	            y=int(y1+i*gap)
	            x=int((x1+x2)/2) 
	            spot_dict[(x1,y,x,y+gap)]=cur_len+1
	            spot_dict[(x,y,x2,y+gap)]=cur_len+2

     
        #cv_show(kp_image,'kp_image')

#print(spot_dict)
#print(tot_spots)

position=image.copy()
for spot in spot_dict.keys():

    (x1,y1,x2,y2)=spot
    cv2.rectangle(position,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)

# cv2.imshow('position', position)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



dictionary = {}
dictionary[0] = 'empty'
dictionary[1] = 'occupied'

model = load_model('car1.h5')

all_car = 0
empty_car = 0
position = image.copy()
for spot in spot_dict.keys():


    (x1,y1,x2,y2)=spot
    (x1,y1,x2,y2)=(int(x1),int(y1),int(x2),int(y2))
    

    #裁剪
    spot_img=image[y1:y2,x1:x2]

    spot_img = cv2.resize(spot_img, (48, 48))
    # cv2.imshow('spot_img', spot_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
  

    
    img = spot_img/255
    img = np.expand_dims(img, axis=0)
    class_predicted = model.predict(img)
    #print(class_predicted)
    ID = np.argmax(class_predicted)
    #print(ID)
    label = dictionary[ID]

    #print(label)
    all_car = all_car+1

    if label == 'empty':
    	cv2.rectangle(position, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), -1)
    	empty_car = empty_car+1

print('总车位：'+str(all_car))
print('空车位：'+str(empty_car))

cv2.putText(position, 'all_car='+str(all_car), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(position, 'empty_car='+str(empty_car), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



cv2.imshow('position', position)
cv2.waitKey(0)
cv2.destroyAllWindows()
  

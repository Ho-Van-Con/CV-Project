import numpy as np
#from scipy.misc.pilutil import imresize
import cv2 
from skimage.feature import hog
from matplotlib import pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn import datasets
from skimage import exposure
import imutils
from imutils import contours

#Khai bao ham cat anh
cropped = {}
def crop(im, height, width, k):
    
    imgwidth = im.shape[1]
    imgheight = im.shape[0]
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im[i: i+height,j: j+width ]
            cropped[str(k)] = a
            img = ~a
            k +=1
    return k

#Khai bao ham xu li pixel de nhan dang
def pixels_to_hog_20(img_array):
    hog_featuresData = []
    for img in img_array:
        fd = hog(img, 
                 orientations=10, 
                 pixels_per_cell=(5,5),
                 cells_per_block=(1,1), 
                 visualize=False)
        hog_featuresData.append(fd)
    hog_features = np.array(hog_featuresData, 'float64')
    return np.float32(hog_features)

#Tao class KNN_MODEL
class KNN_MODEL():
    def __init__(self, k = 3):
        self.k = k
        self.model = cv2.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        retval, results, neigh_resp, dists = self.model.findNearest(samples, self.k)
        return results.ravel()

#Khai bao ham xu li anh, nhan dang anh
def proc_user_img(img_file, model):
    print('loading "%s for digit recognition" ...' % img_file)
    im = cv2.imread(img_file)  
    im = imutils.resize(im,width=300)  
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    #Tien xu li anh 
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(imgray, cv2.MORPH_TOPHAT, rectKernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    thresh = cv2.threshold(gradX, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    #Phat giac duong bien de dong khung 
    _,contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locs = []

    #Loai bo cac vung de chon duoc 4 vung anh co 4 so tren ma the
    for i,c in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        ar = w/float(h)
        if ar > 2.5 and ar < 4.0:
            if (w > 40 and w < 55) and (h > 13 and h < 20):
                locs.append((x, y, w, h))

    #Sap xep cac vung
    locs = sorted(locs, key=lambda x:x[0])
    output = [] 
    k = 0
    u = 0

    #Dua vao toa do duong bien cac vung ta thuc hien nhan dang tung vung
    for i, (gX, gY, gW, gH) in enumerate(locs):
        k = k +1
        groupOutput = []

        imae = cv2.imread("images\Visa.png",0)
        binary = cv2.adaptiveThreshold(imae+9,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,121,9)
        blurred = cv2.medianBlur(binary,1)
        blurred = imutils.resize(blurred, width=300)

        
        if(k==4):
            group =  blurred[gY - 3:gY + gH + 3, gX - 5:gX + gW + 3]
        else:
            group =  blurred[gY - 3:gY + gH + 3, gX - 3:gX + gW + 3]
            
        group2 = cv2.resize(group,(120,60))

        #Thuc hien cat 4 anh trong tung vung 
        u = crop(group2, 60, 30, u)

        #Thuc hien nhan dang tung anh
        for digitROI in range(u-4,u,1):
            im_digit = cropped[str(digitROI)]
            
            im_digit = (255-im_digit)
            im_digit = cv2.resize(im_digit,(57,88))

            hog_img_data = pixels_to_hog_20([im_digit])
            
            kq = model.predict(hog_img_data)
            if (digitROI==14):
                print(str(int(kq[0])))
            string = str(int(kq[0]))

            groupOutput.append(string)

        #Ghi ket qua len cac anh do 
        cv2.rectangle(im,(gX - 5, gY - 5),(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(im, "".join(groupOutput), (gX, gY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        output.extend(groupOutput)

    return im

#Khai bao ham lay so tu anh tham chieu (reference)
def get_digits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    
    return final_bounding_rectangles

#Khai bao ham sap xep ccacso tham chieu trong anh va danh so thu tu
def get_contour_precedence(contour, cols):
    return contour[1] * cols + contour[0]

#Khai bao ham load anh tham chieu va xu li
def load_digits_custom(img_file):
    train_data = []
    train_target = []
    start_class = 1
    im = cv2.imread(img_file)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(imgray,11,255,cv2.THRESH_BINARY_INV)[1]   
       
    _,contours,hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits_rectangles = get_digits(contours,hierarchy)  
    digits_rectangles.sort(key=lambda x:get_contour_precedence(x, im.shape[1]))
     
    for index,rect in enumerate(digits_rectangles):
        x,y,w,h = rect
        cv2.rectangle(im,(x-8,y-8),(x+w+8,y+h+8),(0,255,0),2)
        im_digit = thresh[y:y+h,x:x+w]
        im_digit = (255-im_digit)
        im_digit = cv2.resize(im_digit,(57,88))
        train_data.append(im_digit)
        train_target.append(start_class%10)

        if index>0 and (index+1) % 10 == 0:
            start_class += 1
    
    return np.array(train_data), np.array(train_target)

#------------------chuan bi du lieu--------------------------------------------

#Anh tham chieu va anh nhan dang
TRAIN_IMG = 'images\ocr-reference-2.png'
DETECT_IMG = "images\Visa.png"

digits, labels = load_digits_custom(TRAIN_IMG) 

#In thong tin anh tham chieu 
print('train data shape',digits.shape)
print('train label shape',labels.shape)

#Thuc hien sap xep ngau nhien va chuan bi du lieu de training
digits, labels = shuffle(digits, labels, random_state=256) #Xao tron du lieu
train_digits_data = pixels_to_hog_20(digits)
X_train, X_test, y_train, y_test = train_test_split(train_digits_data, labels, test_size=0.7)

#------------------training va nhan dang anh----------------------------------------

#Thuc hien training va in ket qua % do chinh xac
model = KNN_MODEL(k = 3)
model.train(X_train, y_train)
preds = model.predict(X_test)
print('Accuracy: ',accuracy_score(y_test, preds))

#Thuc nhan dang anh 
model = KNN_MODEL(k = 5)
model.train(train_digits_data, labels)
im = proc_user_img(DETECT_IMG, model)

#------------------xuat ket qua----------------------------------------

#Xuat ket qua ra dang anh va imshow
cv2.imwrite("results\Ket_qua_visa.png",im)
cv2.namedWindow("Ket_qua_visa",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Ket_qua_visa", im)

#Xuat tung so truoc khi nhan dang ra plot show
titles = {}
photo = {}
for so in range(0,16):		
    titles[str(so)] = str(so)
    photo[str(so)] = cropped[str(so)]

for i in range(0,16):
	plot.subplot(4,4,i+1), plot.imshow(photo[str(i)],'gray')
	plot.title(titles[str(i)])
	plot.xticks([]), plot.yticks([])
plot.show()
    
cv2.waitKey()           




import numpy as np
import cv2
from utils import loadImages
from sklearn.model_selection import train_test_split


def computeHOG(images, hogList, size=(64, 64)):
    """
    This function computes the Histogram of Oriented Gradients (HOG) of each
    image from the dataset.
    """

    # Set parameters for HOG descriptor.
    winSize = size
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    
    # Create the HOG descriptor and detector with parameters.
    hog = cv2.HOGDescriptor(
        winSize,
        blockSize,
        blockStride,
        cellSize,
        nbins)

    # Read all images from the image list.
    for image in images:

        # Height and width
        h, w = image.shape[:2]
        
        # Check size
        if w >= size[0] and h >= size[1]:

            #region of interest
            y = (h - size[1]) // 2
            x = (w - size[0]) // 2
            roi = image[y:y + size[1], x:x + size[0]].copy()

            # Compute HOG
            hogList.append(hog.compute(roi))

    return hogList

class SVMdetector():
    """
    Classifier using SVM to detect cars in images.
    """

    def trainModel(self, hogList, labels):
        '''
        Train SVM model from HOG features and corrospomding labels.
        '''
        # Create an empty SVM model.
        svm = cv2.ml.SVM_create()

        # Define the SVM parameters.
        # By default, Dalal and Triggs (2005) use a soft (C=0.01) linear SVM trained with SVMLight.
        svm.setDegree(3)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
        svm.setTermCriteria(criteria)
        svm.setGamma(0)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setNu(0.5)
        svm.setP(0.1)
        svm.setC(0.01)
        svm.setType(cv2.ml.SVM_EPS_SVR)

        svm.train(np.array(hogList), cv2.ml.ROW_SAMPLE, np.array(labels))

        self.saveModel(svm)

        # Retrieves all the support vectors.
        sv = svm.getSupportVectors()

        # Retrieves the decision function.
        rho, _, _ = svm.getDecisionFunction(0)

        # Transpose the support vectors matrix.
        sv = np.transpose(sv)

        # Returns the feature descriptor.
        feature = np.append(sv, [[-rho]], 0)

        return feature

    def initDetector(self, feature):
        '''
        Init Hog descriptor and detector from feature descriptor.
        '''

        # Set parameters for HOG descriptor.
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9

        # Create the HOG descriptor and detector with parameters.
        hog = cv2.HOGDescriptor(
            winSize,
            blockSize,
            blockStride,
            cellSize,
            nbins)

        hog.setSVMDetector(feature)

        return hog

    def buildingDetection(self, hog, image, weightThreshold=0.5):
        '''
        Draw rectangles around detected cars.
        @param weightThreshold: Threshold to remove rectangles below value. 
        High weights indicate a sample classifier with a larger confidence.
        https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
        '''
        winStride = (16, 16)
        padding = (8, 8)
        scale = 1.1
        meanShift = False

        rectangles, weights = hog.detectMultiScale(
            image,
            winStride=winStride,
            padding=padding,
            scale=scale,
            useMeanshiftGrouping=meanShift)

        filtered_rectangles = []
        filtered_weights = []
        for rect, weight in zip(rectangles, weights):
            if weight > weightThreshold:
                (x, y, w, h) = rect
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0))
                
                filtered_rectangles.append(rect)
                filtered_weights.append(weight)

        return filtered_rectangles, filtered_weights

    def loadModel(self):
        svm_model = cv2.ml.SVM_load("./outputs/svm_model.temp")
        return svm_model

    def saveModel(self, model):
        model.save("./outputs/svm_model.temp")
    
if __name__ == '__main__':

    positive_samples = np.array(loadImages('outputs/images/buildings.npy'))
    negative_samples = np.array(loadImages('outputs/images/non-buildings.npy')[:1700])

    # positive_hog = computeHOG(positive_samples)
    # negative_hog = computeHOG(negative_samples)
    dataset = np.concatenate((positive_samples, negative_samples), axis=0)

    # Create the class labels, i.e. (+1) positive and (-1) negative.
    labels = []
    [labels.append(+1) for _ in range(len(positive_samples))]
    [labels.append(-1) for _ in range(len(negative_samples))]

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=42)

    # ========================#
    #  ===  TRAIN SVM   ===   #
    # ========================#

    hogList = []

    # Instanstiate SVM detector
    SVM = SVMdetector()

    # Compute HOG, train and save SVM model
    computeHOG(X_train, hogList)
    feature = SVM.trainModel(hogList, y_train)
    hog = SVM.initDetector(feature)

    path_img = "inputs/data/train/images/austin2.tif"
    img = cv2.imread(path_img)
    # Show the input image in a OpenCV window.
    
    SVM.buildingDetection(hog, img)
    
    cv2.imshow("Luftfoto2", img)
    cv2.imwrite("outputs/svm_classification.tif", img)

    cv2.waitKey(0)

    # When everything done, release the OpenCV window.
    cv2.destroyAllWindows()





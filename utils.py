import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.feature import hog
from glob import glob
import random

#<!--------------------------------------------------------------------------->
#<!--                               DATA UTILS                              -->
#<!--------------------------------------------------------------------------->  

def extractBuildings(img, mask, size=(64,64) , buffer=10):
    """
    Given a binary building mask and an image this function returns 
    list of images created by taking the minimum bounding rectangle
    and resizing each building in a aerial image.
    :param size: image patch size
    :param buffer: buffer added to bounding box
    """
    buildings = [] 

    # Extracting contours from binary image
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # Create copy for cropping
    copy = img.copy()

    # Extracting buildings from the image
    for contour in contours:
        building = copy
        x, y, w, h = cv2.boundingRect(contour)
        building = building[y-buffer : y+h+buffer , x-buffer : x+w+buffer]
        # check if x, y dimension is > 0 in order to resize
        if (building.shape[0] > 0 and building.shape[1] > 0):
            building = cv2.resize(building, size)
            buildings.append(building)
        
    return buildings

def loadDataset(dataset):
    """
    This function load all images from a dataset and return a list of Numpy images.
    """
    # List of images.
    images = []

    # Read all filenames from the dataset.
    for filename in dataset:
        # Read the input image.
        image = cv2.imread(filename)

        # Add the current image on the list.
        if image is not None:
            images.append(image)

    # Return the images list.
    return images

def sampleNegativeImages(images, negativeSample, size=(64, 64), N=500):
    """
    The dataset has several images of high resolution aerial images,
    i.e. called here as negative images. This function select "N" 64x64 negative
    sub-images randomly from each original negative image.
    """
    # Initialize internal state of the random number generator.
    random.seed(1)

    # Final image resolution.
    w, h = size[0], size[1]

    # Read all images from the negative list.
    for image in images:
        # extract random samples from image
        for j in range(N):
            y = int(random.random() * (len(image) - h))
            x = int(random.random() * (len(image[0]) - w))
            sample = image[y:y + h, x:x + w].copy()
            negativeSample.append(sample)

    return negativeSample

def saveImages(img_list, path):
    np.save(path, img_list)

def loadImages(path):
    return np.load(path)

def computeHOG(images):
    hogList = []
    for image in images:
        greyscale = rgb2gray(image)
        hg = hog(
            greyscale,
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(3, 3) 
        )
        hogList.append(hg) 
    return np.array(hogList)

def evaluationMetrics(git):
    '''
    Calculates precision, recall and accuracy given the TP, FP, TN and FN values
    :return: precision, recall, accuracy
    '''
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, accuracy

if __name__ == '__main__':


    # Negative samples
    negativePaths = glob("inputs/data/negative/*.tif")
    negativeImages = loadDataset(negativePaths)
    negativeSamples = []
    sampleNegativeImages(negativeImages, negativeSamples, N=1000)

    # Positive samples
    path_img = "inputs/data/train/images/austin2.tif"
    path_mask = "inputs/data/train/gt/austin2.tif"
    img = cv2.imread(path_img)
    mask = cv2.imread(path_mask, 0)
    
    buildings = extractBuildings(img, mask)

    save_path = 'outputs/images/buildings.npy'
    #saveImages(buildings, save_path)
    #buildings = loadImages(save_path)

    # Show the input image in a OpenCV window.
    cv2.imshow("Luftfoto2", buildings[200])

    cv2.waitKey(0)

    # When everything done, release the OpenCV window.
    cv2.destroyAllWindows()
import cv2
import numpy as np

def extractBuildings(img, mask, size=(60,60) , buffer=3):
    """
    Given a binary building mask and an image this function returns 
    list of images created by taking the minimum bounding rectangle
    and resizing each building in a aerial image.
    :param size: image patch size
    :param buffer: buffer added to bounding box
    """
    buildings = [] 

    # Create a binary image.
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Extracting contours from binary image
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # Create copy for cropping
    copy = img.copy()

    # Extracting buildings from the image
    for contour in contours:
        img = copy
        x, y, w, h = cv2.boundingRect(contour)
        building = img[y-buffer : y+h+buffer , x-buffer : x+w+buffer]
        # check if x, y dimension is > o in order to resize
        if (building.shape[0] > 0 and building.shape[1] > 0):
            building = cv2.resize(building, size)
            buildings.append(building)
        
    return buildings

def saveBuildings(img_list, path):
    np.save(path, img_list)

def loadBuildings(path):
    return np.load(path)

if __name__ == '__main__':

    path_img = "inputs/data/train/images/austin2.tif"
    path_mask = "inputs/data/train/gt/austin2.tif"

    img = cv2.imread(path_img)
    mask = cv2.imread(path_mask, 0)
    
    buildings = extractBuildings(img, mask)

    save_path = 'outputs/images/buildings.npy'
    #saveBuildings(buildings, save_path)
    buildings = loadBuildings(save_path)

    # Show the input image in a OpenCV window.
    cv2.imshow("Luftfoto", buildings[202])

    cv2.waitKey(0)

    # When everything done, release the OpenCV window.
    cv2.destroyAllWindows()
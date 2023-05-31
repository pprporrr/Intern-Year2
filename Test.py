from keras.models import load_model 
from PIL import Image, ImageOps
import numpy as np
import PIL
import os

def extractGroundTruthClass(imagePath):
    class_name = imagePath.split("/")[ -2]
    return class_name

def preprocess(imagePath):
    image = Image.open(imagePath).convert("RGB")
    image = ImageOps.fit(image, (640, 640), Image.Resampling.LANCZOS)
    imageArray = np.asarray(image)
    normalizedImgArray = (imageArray.astype(np.float32) / 127.5) - 1
    return normalizedImgArray

def predictClass(imagePath):
    data = np.ndarray(shape = (1, 640, 640, 3), dtype = np.float32)
    data[0] = preprocess(imagePath)

    prediction = model.predict(data)
    index = np.argmax(prediction)
    className = classNames[index]
    confidenceScore = prediction[0][index]
    
    return className, confidenceScore

if __name__ == "__main__":
    np.set_printoptions(suppress = True)
    
    model = load_model("/Users/ppr/Desktop/Project/Intern-Year2/Model/scratchModel(0.5).h5", compile = False)
    classNames = open("labels.txt", "r").readlines()
    
    testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Coal-Ash_Corrosion (FC-CA)"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Dissimilar_Metal_Weld (SR-DM)"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Erosion_Damage"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Fireside_Corrosion"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Fly_Ash_Erosion (ER-FA)"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Long_Term_Overheat_Damage"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Long_Term_Overheating (SR-LT)"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Oxygen_Corrosion"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Short_Term_Overheat_Damage"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Soot_Blower_Erosion (ER-SB)"
    #testDirectory = "/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Welding_Defects"
    
    totalImages = 0
    correctPredictions = 0

    for root, dirs, files in os.walk(testDirectory):
        for file in files:
            imagePath = os.path.join(root, file)
            
            try:
                Image.open(imagePath)
            except (OSError, PIL.UnidentifiedImageError):
                continue
            
            groundTruthClass = extractGroundTruthClass(imagePath)
            
            className, confidenceScore = predictClass(imagePath)
            
            print("Image:", file)
            print("Predicted Class:", className[2:], end = "")
            print("Confidence Score:", confidenceScore)
            print()
            
            if className[2:].strip() == groundTruthClass.strip():
                correctPredictions += 1
            totalImages += 1
    accuracy = correctPredictions / totalImages * 100
    print("Accuracy: {:.2f}%".format(accuracy)) #Accuracy: 92.48%
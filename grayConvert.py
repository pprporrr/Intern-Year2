import os, cv2

inputDirectory = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/val/Welding_Defects'
outputResize = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/val/Welding_Defects'
#outputGray = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayDatasetsResizeAugmented/test/Coal-Ash_Corrosion (FC-CA)'
#outputGrayResize = '/Users/ppr/Desktop/Project/Intern-Year2/Datasets/GrayscaleResize/Welding/Welding_Defects(LQ-WD)'

count = 1

for filename in os.listdir(inputDirectory):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG') or filename.endswith('.bmp'):
        imagePath = os.path.join(inputDirectory, filename)
        image = cv2.imread(imagePath)

        oriResize = cv2.resize(image, (640, 640))

        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #grayResize = cv2.cvtColor(oriResize, cv2.COLOR_BGR2GRAY)
        
        name = f"Welding_{count}"
        
        outputPathResize = os.path.join(outputResize, name + '.jpg')
        cv2.imwrite(outputPathResize, oriResize)
        
        #outputPathGray = os.path.join(outputGray, name + '.jpg')
        #cv2.imwrite(outputPathGray, gray)
        
        #outputPathGrayResize = os.path.join(outputGrayResize, name + '.jpg')
        #cv2.imwrite(outputPathGrayResize, grayResize)
        count += 1

print("Conversion complete.")
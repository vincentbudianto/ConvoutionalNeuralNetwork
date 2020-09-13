from PIL import Image
import numpy

def extractImage(imgname):
    im = Image.open(imgname)
    original_width, original_height = im.size

    if im.mode == "RGBA":
        return (convertImage(imgname).tolist(), original_width, original_height)
    elif im.mode == "RGB":
        return (numpy.array(im).tolist(), original_width, original_height)
    else:
        return (None, 0, 0)

def convertImage(image_file):
    im = Image.open(image_file)

    np_image = numpy.array(im)
    new_image = numpy.zeros((np_image.shape[0], np_image.shape[1], 3)) 

    for each_channel in range(3):
        new_image[:,:,each_channel] = np_image[:,:,each_channel]  

    # flushing
    np_image = []
    return new_image

if __name__ == "__main__":

    print(extractImage("testo.jpg"))
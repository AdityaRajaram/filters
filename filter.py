# import the necessary packages
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import tkinter.messagebox as tmsg
import cv2
path="/home/aditya/Desktop/filters/Aditya R H.jpg"
panelA=None
panelB=None
panelC=None
panelD=None
converted=None



def help1():
    tmsg.showerror("Image Not Selected","Select an image")

def filename():
    global f
    f=askopenfilename(title="Select Your Image",filetypes=[("Image files","*.png"),("Image files","*.jpg"),("Image files","*.jpeg")])
    if f== "":
        f=None

        

def getfilepath():
    return f


def save1():
    file_name=asksaveasfilename(initialdir = "/home/aditya/Desktop/filters",title = "Select file",filetypes = (('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')))
    if file_name==None:
        return
    else:
        converted.save(file_name)

def printimage(image,edged):
    global panelB,panelA,panelC,panelD
    ph1 = ImageTk.PhotoImage(image)
    ph2 = ImageTk.PhotoImage(edged)
    if panelA is None or panelB is None:
            #panelC=Label(text="original image")
            #panelC.pack(side="top",padx=0)
            panelA = Label(text="Original Image",image=ph1,compound="center")
            panelA.image = ph1
            panelA.pack(side="left", padx=10, pady=20)
            #panelD=Label(text="Filtered image")
            #panelD.pack(side="top")
            panelB = Label(text="Filtered Image",image=ph2,compound="center")
            panelB.image = ph2
            panelB.pack(side="right", padx=10, pady=20)
    else:

        panelA.configure(image=ph1)
        panelB.configure(image=ph2)
        panelA.image = ph1
        panelB.image = ph2


class PencilSketch:
    """Pencil sketch effect
        A class that applies a pencil sketch effect to an image.
        The processed image is overlayed over a background image for visual
        effect.
    """

    def __init__(self,bg_gray='home/aditya/Desktop/filters/pencilsketch_bg.jpg'):
        """Initialize parameters
            :param (width, height): Image size.
            :param bg_gray: Optional background image to improve the illusion
                            that the pencil sketch was drawn on a canvas.
        """
        self.width = 964
        self.height = 964

        # try to open background canvas (if it exists)
        self.canvas = cv2.imread(bg_gray, cv2.CV_8UC1)
        if self.canvas is not None:
            self.canvas = cv2.resize(self.canvas, (self.width, self.height))

    def render(self):
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        # print(getfilepath())
        if f=="":
            help1()
        else:
            img_rgb=cv2.imread(f)
            img_rgb=cv2.resize(img_rgb,(600,300))

            img_gray = cv2.cvtColor((img_rgb), cv2.COLOR_RGB2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)
            img_blend = cv2.divide(img_gray, img_blur, scale=256)

        # if available, blend with background canvas
            if self.canvas is not None:
                img_blend = cv2.multiply(img_blend, self.canvas, scale=1. / 256)
            img_blend=cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
            edged=Image.fromarray(img_blend)
            global converted
            converted=edged
            image=Image.fromarray(img_rgb)
            printimage(image,edged)
        
        
class WarmingFilter:
    """Warming filter
        A class that applies a warming filter to an image.
        The class uses curve filters to manipulate the perceived color
        temparature of an image. The warming filter will shift the image's
        color spectrum towards red, away from blue.
    """

    def __init__(self):
        """Initialize look-up table for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])
    def render(self):
        """Applies warming filter to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """
        # warming filter: increase red, decrease blue

        if f=="":
            help1()
        else:
            global converted
            real_image=cv2.imread(f)
            real_image=cv2.resize(real_image,(600,300))
            img_rgb=real_image
            c_r, c_g, c_b = cv2.split(img_rgb)
            c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
            c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
            img_rgb = cv2.merge((c_r, c_g, c_b))

            # increase color saturation
            c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
            c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

            converted= cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
            real_image=Image.fromarray(real_image)
            converted=Image.fromarray(converted)
            printimage(real_image,converted)

    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))


class CoolingFilter:
    """Cooling filter
        A class that applies a cooling filter to an image.
        The class uses curve filters to manipulate the perceived color
        temparature of an image. The warming filter will shift the image's
        color spectrum towards blue, away from red.
    """

    def __init__(self):
        """Initialize look-up table for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        self.incr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 70, 140, 210, 256])
        self.decr_ch_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                 [0, 30,  80, 120, 192])

    def render(self):
        """Applies pencil sketch effect to an RGB image
            :param img_rgb: RGB image to be processed
            :returns: Processed RGB image
        """

        if f=="":
            help1()
        else:
            global converted

            real_image=cv2.imread(f)
            real_image=cv2.resize(real_image,(600,300))
            img_rgb=real_image
            # cooling filter: increase blue, decrease red
            c_r, c_g, c_b = cv2.split(img_rgb)
            c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
            c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
            img_rgb = cv2.merge((c_r, c_g, c_b))

            # decrease color saturation
            c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
            c_s = cv2.LUT(c_s, self.decr_ch_lut).astype(np.uint8)
            converted=cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
            real_image=Image.fromarray(real_image)
            converted=Image.fromarray(converted)
            printimage(real_image,converted)


    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))



class Cartoonizer:
    """Cartoonizer effect
        A class that applies a cartoon effect to an image.
        The class uses a bilateral filter and adaptive thresholding to create
        a cartoon effect.
    """

    def __init__(self):
        pass

    def render(self):
        if f=="":
            help1()
        else:
            global converted

            real_image=cv2.imread(f)
            real_image=cv2.resize(real_image,(600,300))
            img_rgb=real_image
            numDownSamples = 2       # number of downscaling steps
            numBilateralFilters = 7  # number of bilateral filtering steps

            # -- STEP 1 --
            # downsample image using Gaussian pyramid
            img_color = img_rgb
            for _ in range(numDownSamples):
                img_color = cv2.pyrDown(img_color)

            # repeatedly apply small bilateral filter instead of applying
            # one large filter
            for _ in range(numBilateralFilters):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 9)

            # upsample image to original size
            for _ in range(numDownSamples):
                img_color = cv2.pyrUp(img_color)

            # make sure resulting image has the same dims as original
            img_color = cv2.resize(img_color, img_rgb.shape[:2])

            # -- STEPS 2 and 3 --
            # convert to grayscale and apply median blur
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            img_blur = cv2.medianBlur(img_gray, 7)

            # -- STEP 4 --
            # detect and enhance edges
            img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY, 9, 11)

            # -- STEP 5 --
            # convert back to color so that it can be bit-ANDed with color image
            img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
            width=img_edge.shape[1]
            height=img_edge.shape[0]
            img_color=cv2.resize(img_color,(width,height))
            print(img_color.shape)
            print(img_edge.shape)
            cv2.bitwise_and(img_color, img_edge,img_edge)
            real_image=Image.fromarray(real_image)
            converted=Image.fromarray(img_edge)
            printimage(real_image,converted)


# initializ


f=""

# initialize the window toolkit along with the two image panels
root = Tk()
root.geometry("600x400")
root.title("Different types of filters")
root.configure(bg="orange red")
root.iconphoto(True, PhotoImage(file="/home/aditya/Desktop/filters/filter.png"))
text=Label(root,text="Welcome",font="Helvetica 40 bold underline",bg="orange red",fg="gold",pady=30)
text.pack()
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI



pencil=PencilSketch()
warm=WarmingFilter()
cool=CoolingFilter()
cartoon=Cartoonizer()
btn = Button(root, text="Select an image", command=filename,width=150,relief=SUNKEN,fg="black",bg="white")
btn.pack(padx="10", pady="25")

f1=Frame(root,bg="orange red")
f1.pack(pady=25)
btn1 = Button(f1, text="apply for pencil sketch", command=pencil.render,relief=SUNKEN,fg="black",bg="white")
btn1.grid(row=15,column=3,padx=10)
btn2 = Button(f1, text="apply for warm filter", command=warm.render,relief=SUNKEN,fg="black",bg="white")
btn2.grid(row=15,column=6,padx=10)
btn3 = Button(f1, text="apply for cool filter", command=cool.render,relief=SUNKEN,fg="black",bg="white")
btn3.grid(row=15,column=9,padx=10)
btn4 = Button(f1, text="apply for cartoonizing filter", command=cartoon.render,relief=SUNKEN,fg="black",bg="white")
btn4.grid(row=15,column=12,padx=10)

btn5=Button(root,text="save",command=save1,relief=SUNKEN,fg="black",bg="white")
btn5.pack(side="bottom",padx=15,pady=15)




    


# kick off the GUI

root.mainloop()




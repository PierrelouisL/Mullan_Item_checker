import tkinter
from tkinter import *
from tkinter.font import Font
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import pyodbc
import os
from datetime import datetime
import requests

def get_parent_dir(n=1):
    """returns the n-th parent directory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

import sys
import argparse
from keras_yolo3.yolo import YOLO, detect_video, detect_webcam
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object
import test
import utils
import pandas as pd
import numpy as np
from Get_File_Paths import GetFileList
import random
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None

# define YOLO detector
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": 0.25,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)
# Make a dataframe for the prediction outputs
out_df = pd.DataFrame(
    columns=[
        "image",
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
        "confidence",
        "x_size",
        "y_size",
    ]
)

"""         READ ME TO UNDERSTAND model_cl_fill function and several lists and dict:
    model_classes_nb dictionnary is to store the amount of items we have, the key is the item name
    model_nb is the list to store which class correspond to class 1, class 2... (the numbers who are used to return prediction)
    model_cl_fill function resets the dict to zero so the items dont add up each time we predict."""

model_classes_nb = {}
model_nb = []
items_db = {}
order_items = {}

file1 = open(model_classes, 'r')
while True:
    line = file1.readline()
    if not line:
        break
    model_nb.append(line.strip())
file1.close()

def model_cl_fill():
    global model_classes_nb
    for class_nb in range(len(model_nb)):
        model_classes_nb[model_nb[class_nb]] = 0
model_cl_fill()


def predict_img(frame):
    global out_df
    global yolo
    global model_classes_nb
    cv2.imwrite(os.path.join(image_test_folder, 'Frame_app_live.jpg'), frame)  # Temporarly writes an img to predict it afterwards
    start_t = timer()

    # This is for the image we just taken
    img_path = os.path.join(image_test_folder, 'Frame_app_live.jpg')
    prediction, image = detect_object(
        yolo,
        img_path,
        save_img=True,
        save_img_path=detection_results_folder,
        postfix="_catface",
    )
    y_size, x_size, _ = np.array(image).shape
    model_cl_fill()
    for single_prediction in prediction:  # For each prediction we add them to the array of predictions and draw them
        out_df = out_df.append(
            pd.DataFrame(
                [
                    [
                        os.path.basename(img_path.rstrip("\n")),
                        img_path.rstrip("\n"),
                    ]
                    + single_prediction
                    + [x_size, y_size]
                ],
                columns=[
                    "image",
                    "image_path",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "label",
                    "confidence",
                    "x_size",
                    "y_size",
                ],
            )
        )
        model_classes_nb[model_nb[single_prediction[4]]] += 1
    end = timer()
    print(
        "Processed 1 images in {:.1f}sec - {:.1f}FPS".format(
            end - start_t,
            1 / (end - start_t),
        )
    )
    out_df.to_csv(detection_results_file, index=False)


def view():
    """ Here is an example of what does the function view returns:
    {0: (4, '56629', 'ACCU-100X2-BRASS.POLISHED', 'Brass Washer 100mm x 2mm | 10.5mm Center Hole'), 1: (0, '56629', 'STER-BS2871-CZ108', '1/2" Half Hard seamless brass tube x 1mtr'), ... }
    It is a list of every needed items: {item_nb: ( qty, 'SalesOrderNb', 'Item name', 'Item Desc'), ...}
    """
    global salesorder
    cnxn = None
    server = '192.168.XX.XX,XXXX'
    database = 'XXXXXX'
    username = 'XXXXXX'
    password = 'XXXXXX'
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    cursor.execute(
        "SELECT l.[Quantity],LEFT(w.[WONumber],5) as WONumber,s.[Code],s.[name] FROM [MULLANIE].[dbo].[SiWorksOrder] as w LEFT JOIN [MULLANIE].[dbo].[SiWorksOrderLine] as l ON w.[SiWorksOrderID] = l.[SiWorksOrderID] LEFT JOIN [MULLANIE].[dbo].[StockItem] as s ON l.[ItemID] = s.[ItemID] LEFT JOIN [unio].[dbo].[outstock] u  ON u.code = s.Code LEFT JOIN [MULLANIE].[dbo].[BinItem] b ON b.[ItemID] = s.[ItemID] LEFT JOIN [MULLANIE].[dbo].[SOPOrderReturnLine] c ON c.[SOPOrderReturnLineID] = w.[SOPOrderReturnLineID] LEFT JOIN [MULLANIE].[dbo].[SOPOrderReturn] z ON z.SOPOrderReturnID = c.SOPOrderReturnID LEFT JOIN [MULLANIE].[dbo].[ProductGroup] p ON s.ProductGroupID = p.ProductGroupID WHERE w.[WONumber]= ? AND w.[WOStatus] NOT IN ('Deleted') ORDER BY p.Code DESC",
        salesorder)
    global items_db
    items_db= {}
    nb = 0
    for row in cursor.fetchall():
        items_db[nb] = (int(row[0]), row[1], row[2], row[3])
        nb += 1


def get_all_lights():
    """ Function to get all lights from a sales order number: returns 1 or 0 if it exists or not.
    Everything is stored in order_item[SO-00x] = (Name of the light , Desc) the name is for example MLWL413PCWTE and x is the item number"""
    global salesordern
    global order_items
    cnxn = None
    server = '192.168.XX.XX,XXXX'
    database = 'XXXXXX'
    username = 'XXXXXX'
    password = 'XXXXXX'
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    cursor.execute(
        "SELECT  e.Code,ItemDescription as Name,WONumber FROM [MULLANIE].[dbo].[SiWorksOrder] a LEFT JOIN [MULLANIE].[dbo].[StockItem] as e ON [StockItemID] = e.ItemID LEFT JOIN [MULLANIE].[dbo].[SOPOrderReturnLine] c  ON c.[SOPOrderReturnLineID] = a.[SOPOrderReturnLineID] LEFT JOIN [MULLANIE].[dbo].[SOPOrderReturn] d ON c.[SOPOrderReturnID] = d.[SOPOrderReturnID] LEFT JOIN [MULLANIE].[dbo].[SLCustomerAccount] f ON [CustomerID] = f.[SLCustomerAccountID] LEFT JOIN [MULLANIE].[dbo].[SLCustomerLocation] g ON f.[SLCustomerAccountID] = g.[SLCustomerAccountID] LEFT JOIN [MULLANIE].[dbo].[SYSCountryCode] h ON g.[AddressCountryID] = h.[SYSCountryCodeID] LEFT JOIN [MULLANIE].[dbo].[SOPDocDelAddress] i ON c.[SOPOrderReturnID] = i.[SOPOrderReturnID] WHERE [DocumentNo] = ? AND [WOStatus] NOT IN ('Deleted')",
        salesordern)
    order_items = {}
    for row in cursor.fetchall():
        order_items[row[2]] = (row[0], row[1])
    if not len(order_items):
        tkinter.messagebox.showerror(title="Error", message="The Sales Order number you selected doesnt exist on the Data Base!")
        return 0
    else:
        return 1


def update_lbox():
    """Here we will update the listbox using the items_db dictionnary (to have every info about the item) and tell how many of the items we detected"""
    global items_db  # item dict from database
    global model_classes_nb  # items dict we detected
    total_detected = 0
    total_needed = 0
    lbox.delete(0, END)
    # Here we get every item that we got from DB then we compare with the items we detected to see how many of those match and print red if not enough or green if enough
    # example: items[1] = (4, '56629', 'ACCU-100X2-BRASS.POLISHED', 'Brass Washer 100mm x 2mm | 10.5mm Center Hole')
    #                   Qty | Order number |    item type         |                item desc
    # model_classes_nb is the items we detected
    for items in items_db.items():
        try:
            #foo = model_classes_nb[items[1][2]] # TO test if the item is in the IA MODEL! Disabled for now because small model
            model_classes_nb[items[1][2]] = 0  # For now because we dont have the model required
            if model_classes_nb[items[1][2]] <= items[1][0] and items[1][0] >= 1:
                # We have enough items! Print in green
                lbox_string = items[1][2] + ": ( " + str(items[1][0]) + " | " + str(model_classes_nb[items[1][2]]) + ")"
                lbox.insert(tkinter.END, lbox_string)
                lbox.itemconfig(tkinter.END, bg='red', fg='white')
            else:
                # We do not have enough items print in red
                lbox_string = items[1][2] + ": ( " + str(items[1][0]) + " | " + str(model_classes_nb[items[1][2]]) + ")"
                lbox.insert(tkinter.END, lbox_string)
                lbox.itemconfig(tkinter.END, bg='green', fg='white')
        except KeyError:
            lbox.insert(tkinter.END, items[1][2] + " -> NOT IN IA MODEL")
            lbox.itemconfig(tkinter.END, bg='YELLOW', fg='black')
        total_needed += items[1][0]
        total_detected += model_classes_nb[items[1][2]]
        if total_needed > total_detected:
            Summary.configure(text=str(total_needed)+" / "+str(total_detected), bg='red')
        else:
            Summary.configure(text=str(total_needed) + " / " + str(total_detected), bg='green')


def start():
    """Function to read camera and to analyse or not"""
    view()
    global analyse
    global cap
    cap = cv2.VideoCapture(1)
    analyse = 1 - analyse

    def show_frame():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        global analyse
        if analyse == 0:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((int(2.5*boxwidth), int(2.5*boxheight)), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)
        else:
            # Analyse == 1 so we need to analyse the picture we have just taken
            print("Predicting new frame...")
            predict_img(frame)
            # print(os.path.join(detection_results_folder, 'Frame_nb_%d_catface' %img_id + ".jpg"))
            result_im = cv2.imread(os.path.join(detection_results_folder, 'Frame_app_live_catface.jpg'))
            print("The items detected are:", end='')
            for name in model_classes_nb.keys():
                print(" {} {} - ".format(name, model_classes_nb[name]), end='')
            print('')
            cv2image = cv2.cvtColor(result_im, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            img = img.resize((int(2.5 * boxwidth), int(2.5 * boxheight)), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
    show_frame()
    update_lbox()


class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.master = master
        pad = 3
        self._geom = '980x540+{0}+{1}'.format(master.winfo_screenwidth()/2, master.winfo_screenheight()/2)
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth() - pad, master.winfo_screenheight() - pad))
        master.bind('<Escape>', self.toggle_geom)
        #master.bind('<Configure>', self.resize_ev) TODO: do the resize window event

    def toggle_geom(self, event):
        geom = self.master.winfo_geometry()
        print(geom, self._geom)
        self.master.geometry(self._geom)
        self._geom = geom

    """def resize_ev(self, event):
        print(event.width)
        print(event.height)"""


def create_new_window():
    def close():
        global salesorder
        try:
            item_selected = lbox_items.curselection()[0]
        except IndexError:
            item_selected = 0
        if item_selected < 10:
            item_selected_str = "-00" + str(item_selected+1)  # Sales Order is xxxxx-001, xxxxx-002 ... So we concatenate the item selected with -00
        else:
            item_selected_str = "-0" + str(item_selected+1)
        window.destroy()
        salesorder = salesorder + item_selected_str
        l_order.config(text=salesorder)
        # print(order_items[salesorder][1])  # Prints the description of the item selected (it's already specified in the new SO, reminder: SO = xxxxxx-nb_item)
        stop_at = order_items[salesorder][1].find('\n')  # Some SO descriptions have \n which make them impossible to print on the app so we look for them and show only the first line
        if stop_at > 0:  # because if it detects a \n it returns the index nb and if not "-1"
            new_desc = ['']
            for i in range(stop_at):
                new_desc.append(order_items[salesorder][1][i])
            l_order_desc.config(text="".join(map(str, new_desc)))  # Convert new_desc list to string
        else:
            l_order_desc.config(text=order_items[salesorder][1])
        view()  # To update new components in the list
        update_lbox()  # to update the listbox and show every component

    window = Toplevel(root)  # create the new window
    window.geometry('1000x350+500+150')
    window.grab_set()  # Don't allow user to touch the old window
    fontwin = Font(family="Lucida Grande", size=20)
    Label(window, text="Choose an item you want to check", font=fontwin).pack(side=TOP)
    Button(window, text="OK", font=fontwin, command=close).pack(side=BOTTOM)
    lbox_items = Listbox(window, font=fontwin, yscrollcommand=True)
    lbox_items.pack(expand=True, fill=X)
    lbox_items.xview()
    for key in order_items.keys():
        lbox_items.insert(tkinter.END, order_items[key])
    window.mainloop()


def get():
    global salesorder
    global salesordern
    salesorder = new_SO.get()
    salesordern = "00000" + salesorder
    if get_all_lights():
        create_new_window()


if __name__ == "__main__":
    global analyse
    global salesorder
    global salesordern
    salesordern = "0000056629"
    salesorder = "56629-001"
    analyse = 1
    root = Tk()
    app = FullScreenApp(root)
    fontStyle = Font(family="Lucida Grande", size=20)
    VeryBigFont = Font(family="Lucida Grande", size=50)
    root.title("Mullan's item checker")

    boxheight = root.winfo_screenheight()/3
    boxwidth = root.winfo_screenwidth()/4

    Label(root, text="Items Summary: \n (Items needed | Items detected)", font=fontStyle).place(x=1.5*boxwidth, y=0, height=0.5*boxheight, width=1.5*boxwidth)
    Summary = Label(root, font=VeryBigFont, justify=RIGHT)
    Summary.place(x=3*boxwidth, y=0, height=0.5*boxheight, width=boxwidth)

    lmain = Label(root)
    lmain.place(x=1.5*boxwidth+20, y=0.5*boxheight, height=2.5*boxheight, width=2.5*boxwidth)

    lbox = Listbox(root, font=fontStyle)
    lbox.place(x=10, y=boxheight, height=2*boxheight, width=1.5*boxwidth)

    Button_refresh = Button(root, text="Analyse / Get camera view", command=start, font=fontStyle)
    Button_refresh.place(x=10, y=boxheight/2, height=0.25*boxheight, width=1.5*boxwidth)

    fontStyle = Font(family="Lucida Grande", size=16)
    lbox_info = Label(root, text="Items: (Number needed | Number detected) - Green if OK, Red if missing", font=fontStyle)
    lbox_info.place(x=10, y=(boxheight/2)+0.25*boxheight, height=0.25*boxheight, width=1.5*boxwidth)

    new_SO = tkinter.StringVar()
    get_all_lights()
    ent = Entry(root, textvariable=new_SO, font=fontStyle, justify=CENTER)
    ent.place(x=10, y=0.25*boxheight, height=0.25*boxheight, width=0.6*boxwidth)
    Button(root, text="New sales order", command=get, font=fontStyle).place(x=0.75*boxwidth+10, y=0.25*boxheight, height=0.25*boxheight, width=0.45*boxwidth)
    fontStyle = Font(family="Lucida Grande", size=25)
    l_order = Label(root, text=salesorder, font=fontStyle)
    l_order.place(bordermode=OUTSIDE, height=0.125 * boxheight, width=1.5 * boxwidth)
    l_order_desc = Label(root, text=order_items[salesorder][1], font=Font(family="Lucida Grande", size=15))
    l_order_desc.place(x=0, y=0.125*boxheight, height=0.125*boxheight, width=1.5*boxwidth)

    # Start the Camera for the first time
    start()

    root.mainloop()

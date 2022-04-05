import glob
import argparse
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

import architectures as archs
import parameters as par
import torch
import torch.nn as nn
import torchvision.transforms as transforms

####################################################################
parser = argparse.ArgumentParser()
parser = par.basic_training_parameters(parser)
parser = par.batch_creation_parameters(parser)
parser = par.batchmining_specific_parameters(parser)
parser = par.loss_specific_parameters(parser)
opt = parser.parse_args()
####################################################################

class demo(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack() 
        self.master.geometry("900x540")
        self.master.title("Compare 2 images")
        self.create_widgets()

    def create_widgets(self):
        ##Left-----------------------------------------------------------------------------
        #food image
        self.frame_canvas1 = Canvas(self, bg='azure3', height=300, width=300)
        self.frame_canvas1.grid(row=0, column=0)

        #list of foods
        self.frame_list1 = Frame(self)
        self.frame_list1.grid(row=1, column=0)
        self.foods = glob.glob("data/**/*")
        v1 = StringVar(value=self.foods)
        self.foods_list1 = Listbox(self.frame_list1, listvariable=v1, height=10, width=50)
        self.foods_list1.grid()
        
        #button
        self.select_btn1 = Button(self,width=42)
        self.select_btn1['text'] = 'select'
        self.select_btn1['command'] = self.select_image1
        self.select_btn1.grid(row=2, column=0)

        #label
        self.selected_food1 = Label(self)
        self.selected_food1['text'] = 'Image is not selected'
        self.selected_food1.grid(row=3, column=0)

        ##Right-----------------------------------------------------------------------------
        #food image
        self.frame_canvas2 = Canvas(self, bg='azure3', height=300, width=300)
        self.frame_canvas2.grid(row=0, column=1)

        #list of foods
        self.frame_list2 = Frame(self)
        self.frame_list2.grid(row=1, column=1)
        self.foods = glob.glob("data/**/*")
        v2 = StringVar(value=self.foods)
        self.foods_list2 = Listbox(self.frame_list2, listvariable=v2, height=10, width=50)
        self.foods_list2.grid()
        
        #button
        self.select_btn2 = Button(self,width=42)
        self.select_btn2['text'] = 'select'
        self.select_btn2['command'] = self.select_image2
        self.select_btn2.grid(row=2, column=1)

        #label
        self.selected_food2 = Label(self)
        self.selected_food2['text'] = 'Image is not selected'
        self.selected_food2.grid(row=3, column=1)

        ##Calculator------------------------------------------------------------------------
        #Button
        self.calc_btn = Button(self,width=20)
        self.calc_btn['text'] = 'calculate similarity'
        self.calc_btn['command'] = self.calc_similarity
        self.calc_btn.grid(row=2, column=2)

        #Result
        self.result = Label(self, font=("Helvetica","13"))
        self.result['text'] = ''
        self.result.grid(row=3, column=2)
        
    ##Process-------------------------------------------------------------------------------

    def select_image1(self):
        itemIdxList = self.foods_list1.curselection()
        self.selected_food1['text'] = self.foods[itemIdxList[0]]
        self.image1 = Image.open(open('{}'.format(self.foods[itemIdxList[0]]),'rb'))
        self.image1.thumbnail((300, 300))
        self.image1tk = ImageTk.PhotoImage(self.image1)
        self.frame_canvas1.create_image(
            0,
            0,
            image=self.image1tk,
            anchor=NW,
        )
    def select_image2(self):
        itemIdxList = self.foods_list2.curselection()
        self.selected_food2['text'] = self.foods[itemIdxList[0]]
        self.image2 = Image.open(open('{}'.format(self.foods[itemIdxList[0]]),'rb'))
        self.image2.thumbnail((300, 300))
        self.image2tk = ImageTk.PhotoImage(self.image2)
        self.frame_canvas2.create_image(
            0,
            0,
            image=self.image2tk,
            anchor=NW,
        )
    def calc_similarity(self):
        emb1 = embedding(self.image1)
        emb2 = embedding(self.image2)
        similarity = cos_similarity(emb1, emb2)
        self.result['text'] = '{:3f}'.format(float(similarity))


MODEL_PATH = "weight.pth"

model = archs.select("resnet50_frozen", opt)
model = nn.DataParallel(model)
model_params = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(model_params)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]
)

def cos_similarity(v1, v2):
    v2 = v2.T
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def embedding(img):
    input_img = transform(img)
    img_batch = input_img[None]
    out = model(img_batch)
    emb = out[0].cpu().detach().numpy().copy()
    return emb

def main():
    root = Tk()
    app = demo(master=root)
    app.mainloop()

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:40:19 2018

@author: Brendan
"""

import tkinter
from tkinter import messagebox, Checkbutton, IntVar, Label, Entry, LEFT, RIGHT, Tk, Frame, BOTTOM, Button, StringVar
from tkinter import RAISED, Listbox, Menu, Menubutton, Toplevel, Message, Radiobutton, W, DoubleVar, Scale, CENTER, OUTSIDE
from tkinter import Scrollbar, END, BOTH, Y, Text, INSERT, Spinbox, PanedWindow, VERTICAL, LabelFrame, FLAT, SUNKEN, GROOVE, RIDGE, TOP


def helloCallBack():
    messagebox.showinfo( "Hello Python", "Hello World")
    
def helloCallBack2():
    messagebox.showinfo( "Hello World", "Hello Python")
    #print(E1.get())

#top = Tk()    



"""
###
CheckVar1 = IntVar()
CheckVar2 = IntVar()

C1 = Checkbutton(top, text = "Music", variable = CheckVar1, \
onvalue = 1, offvalue = 0, height=5, \
width = 20)

C2 = Checkbutton(top, text = "Video", variable = CheckVar2, \
onvalue = 1, offvalue = 0, height=5, \
width = 20, command = helloCallBack2)

C1.pack()
C2.pack()
"""



"""
CheckVar2 = IntVar()
C2 = Checkbutton(top, text = "Video", variable = CheckVar2, \
onvalue = 1, offvalue = 0, height=5, \
width = 20, command = helloCallBack2)
C2.pack()

### Label/Entry *single line text entry
var = StringVar()
L1 = Label(top, text="User Name")
L1.pack( side = LEFT)
E1 = Entry(top, bd =5, textvariable=var)
E1.pack(side = RIGHT)
"""


"""
### menu with click-dropdown options
mb= Menubutton ( top, text="condiments", relief=RAISED )
mb.grid()
mb.menu = Menu ( mb, tearoff = 0 )
mb["menu"] = mb.menu

mayoVar = IntVar()
ketchVar = IntVar()

mb.menu.add_checkbutton ( label="mayo",
variable=mayoVar )
mb.menu.add_checkbutton ( label="ketchup",
variable=ketchVar, command = helloCallBack )
mb.pack()
"""


"""
### cursor when hover over button
B1 = tkinter.Button(top, text ="circle", relief=RAISED,\
cursor="circle")
B2 = tkinter.Button(top, text ="plus", relief=RAISED,\
cursor="plus")
B1.pack()
B2.pack()
"""

#top.mainloop()

root = Tk()


"""
### editable text -- could use to pass arguments / config file?
def onclick():
    pass
text = Text(root)
text.insert(INSERT, "Hello.....")
text.insert(END, "Bye Bye.....")
text.pack()

text.tag_add("here", "1.0", "1.4")
text.tag_add("start", "1.8", "1.13")
text.tag_config("here", background="yellow", foreground="blue")
text.tag_config("start", background="black", foreground="green")
"""


"""
root = Tk()
top = Toplevel()
top.mainloop()
"""


"""
m1 = PanedWindow()

m1.pack(fill=BOTH, expand=1)
left = Label(m1, text="left pane")
m1.add(left)
m2 = PanedWindow(m1, orient=VERTICAL)
m1.add(m2)
top = Label(m2, text="top pane")
m2.add(top)

bottom = Label(m2, text="bottom pane")
m2.add(bottom)
"""



#"""
###
def donothing():
    filewin = Toplevel(root)
    button = Button(filewin, text="Do nothing button")
    button.pack()
    

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=donothing)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
filemenu.add_command(label="Save as...", command=donothing)
filemenu.add_command(label="Close", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo", command=donothing)
editmenu.add_separator()
editmenu.add_command(label="Cut", command=donothing)
editmenu.add_command(label="Copy", command=donothing)
editmenu.add_command(label="Paste", command=donothing)
editmenu.add_command(label="Delete", command=donothing)
editmenu.add_command(label="Select All", command=donothing)
menubar.add_cascade(label="Edit", menu=editmenu)

helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=donothing)
helpmenu.add_command(label="About...", command=donothing)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)


root.title('Test1')
root.geometry("1000x500")
#root.resizable(0, 0)
topframe = Frame(root)
topframe.pack(side = TOP)
leftframe = Frame(root)
leftframe.pack(side = LEFT)
rightframe = Frame(root)
rightframe.pack(side = RIGHT)
labelframe = LabelFrame(topframe, text="Parameters")
labelframe.pack(fill="both", expand="yes")
redbutton = Button(labelframe, text="Red", fg="red", relief=RAISED)
redbutton.pack( side = LEFT)
greenbutton = Button(labelframe, text="Brown", fg="brown", relief=GROOVE)
greenbutton.pack( side = LEFT )
bluebutton = Button(labelframe, text="Blue", fg="blue", relief=RIDGE)
bluebutton.pack( side = LEFT )


def sel():
    selection = "You selected the option " + str(var.get())
    label.config(text = selection)

var = IntVar()
R1 = Radiobutton(root, text="Option 1", variable=var, value=1,
command=sel)
R1.pack( anchor = W )
R2 = Radiobutton(root, text="Option 2", variable=var, value=2,
command=sel)
R2.pack( anchor = W )

R3 = Radiobutton(root, text="Option 3", variable=var, value=3,
command=sel)
R3.pack( anchor = W)
label = Label(root)
label.pack()


def sel2():
    selection2 = "Value = " + str(var2.get())
    label.config(text = selection2)
    print(int(var2.get()))
    return x
    
global x
x = 0    
var2 = DoubleVar()
scale = Scale( root, variable = var2 )
scale.pack(anchor=CENTER)

button = Button(root, text="Get Scale Value", command=sel2)
button.pack(anchor=CENTER)
label2 = Label(root)
label2.pack()

#"""



"""
for r in range(3):
    for c in range(4):
        tkinter.Label(root, text='R%s/C%s'%(r,c),
                      borderwidth=1 ).grid(row=r,column=c)
"""
root.mainloop()


"""
top = Tk()

B = tkinter.Button(top, text ="Hello", command = helloCallBack)
B.pack()
B.place(bordermode=OUTSIDE, height=100, width=100)
top.mainloop()
"""


### dimensions: integer = pixels, can specify 'c' for centimeters, 'i' for inches
### anchors: NW, N, NE, W, CENTER, E, SW, S, SE


"""
##################################
### probably wont use
##################################
"""

"""
### just some text on a label
var = StringVar()
label = Label( root, textvariable=var, relief=RAISED )

var.set("Hey!? How are you doing?")
label.pack()
"""


"""
### just a multi-line label
var = StringVar()
label = Message( root, textvariable=var, relief=RAISED )
var.set("Hey!? How are you doing?")
label.pack()
"""


"""
### List Box - meh
Lb1 = Listbox(top)
Lb1.insert(1, "Python")
Lb1.insert(2, "Perl")
Lb1.insert(3, "C")
Lb1.insert(4, "PHP")
Lb1.insert(5, "JSP")
Lb1.insert(6, "Ruby")

Lb1.pack()
"""


"""
### scrolling list box
scrollbar = Scrollbar(root)
scrollbar.pack( side = RIGHT, fill=Y )

mylist = Listbox(root, yscrollcommand = scrollbar.set )
for line in range(100):
    mylist.insert(END, "This is line number " + str(line))
    
mylist.pack( side = LEFT, fill = BOTH )
scrollbar.config( command = mylist.yview )
"""


"""
### box where click arrows to increment
w = Spinbox(root, from_=0, to=10)
w.pack()
"""


"""
### bitmap: help / info...
B1 = tkinter.Button(top, text ="error", relief=RAISED,\
bitmap="error")
B2 = tkinter.Button(top, text ="hourglass", relief=RAISED,\
bitmap="hourglass")
B3 = tkinter.Button(top, text ="info", relief=RAISED,\
bitmap="info")
B4 = tkinter.Button(top, text ="question", relief=RAISED,\
bitmap="question")
B5 = tkinter.Button(top, text ="warning", relief=RAISED,\
bitmap="warning")
B1.pack()
B2.pack()
B3.pack()
B4.pack()
B5.pack()
"""


"""
def hello():
    messagebox.showinfo("Say Hello", "Hello World")
B1 = tkinter.Button(root, text = "Say Hello", command = hello)
B1.pack()
"""


"""
### Button
B = tkinter.Button(top, text ="Hello", command = helloCallBack)
B.pack()
C = tkinter.Button(top, text ="World", command = helloCallBack2)
C.pack()
"""


"""
### Canvas
C = tkinter.Canvas(top, bg="blue", height=250, width=300)
oval = C.create_oval(10, 50, 250, 150, fill='green')

C.pack()
"""
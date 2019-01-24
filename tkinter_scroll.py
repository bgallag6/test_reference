# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 18:42:02 2019

@author: Brendan
"""
# how to clear list
# reset button

from tkinter import *
import os
import glob
from tkinter import Frame, Listbox, Scrollbar, Label, Button
from tkinter import END, Tk, N, S, E, W, EXTENDED

def select():
    selected_text_list = [lb_table.get(i) for i in lb_table.curselection()]
    for j in range(len(selected_text_list)):
        print(selected_text_list[j])
        lb_field.insert(END, selected_text_list[j])
        
def select2():
    selected_text_list = [lb_field.get(i) for i in lb_field.curselection()]
    for j in range(len(selected_text_list)):
        print(selected_text_list[j])
        lb_select.insert(END, selected_text_list[j])

def get_list(event):
    #function to read the listbox selection
    #and put the result in an entry widget
    # get selected line index
    index = lb_table.curselection()[0]
    # get the line's text
    seltext = lb_table.get(index)
    # delete previous text in enter1
    lb_field.delete(END)
    # now display the selected text
    lb_field.insert(END, seltext)
    
def get_list3(event):
    #function to read the listbox selection
    #and put the result in an entry widget
    # get selected line index
    index = lb_table.curselection()[0]
    # get the line's text
    seltext = lb_table.get(index-1)
    # delete previous text in enter1
    lb_field.delete(END)
    # now display the selected text
    lb_field.insert(END, seltext)
    
def get_list4(event):
    #function to read the listbox selection
    #and put the result in an entry widget
    # get selected line index
    index = lb_table.curselection()[0]
    # get the line's text
    seltext = lb_table.get(index+1)
    # delete previous text in enter1
    lb_field.delete(END)
    # now display the selected text
    lb_field.insert(END, seltext)
    
def del_select(event):
    # delete selected record
    print(lb_select.curseselection()[0])
    lb_select.delete(lb_select.curselection()[0])  # deletes last entry if click on empyt space
    
    
flist = sorted(glob.glob('S:/FITS/20001010/1600/aia*.fits'))

#"""
root = Tk()
root.title('Database Query')

root.geometry('{}x{}'.format(1100, 800))

table_frame = Frame(root, padx=3, pady=3)
field_frame = Frame(root, padx=3, pady=3)
selected_frame = Frame(root, padx=3, pady=3)
btmrt_frame = Frame(root, padx=3, pady=3)
cen_frame = Frame(root, padx=3, pady=3)

table_frame.grid(row=0)
field_frame.grid(row=1, rowspan=2)
selected_frame.grid(row=0, column=2, rowspan=2)
btmrt_frame.grid(row=3, column=2)
cen_frame.grid(row=0, column=1, rowspan=3)


table_label = Label(table_frame, text='Database Tables')
table_label.grid(row=0)

fields_label = Label(field_frame, text='Table Fields')
fields_label.grid(row=0)

records_label = Label(selected_frame, text='Selected Records')
records_label.grid(row=0)


btn = Button(cen_frame, text="Add >>", command=select, width=7)
btn.grid(row=0, column=0)

btn2 = Button(cen_frame, text="<< Delete", command=select2, width=7)
btn2.grid(row=1, column=0, pady=3)

btn = Button(btmrt_frame, text="Table", command=select)
btn.grid(row=0, column=0)

btn2 = Button(btmrt_frame, text="Fields", command=select2, width=10)
btn2.grid(row=0, column=1)


sb_table = Scrollbar(table_frame)
sb_table.grid(row=1, column=1, sticky=N+S)

#listbox2 = Listbox(table_frame, selectmode=EXTENDED, yscrollcommand=scrollbar2.set, width=80, height=14)  # width in characters, height in lines of text
lb_table = Listbox(table_frame, yscrollcommand=sb_table.set, width=80, height=14)  # width in characters, height in lines of text
for i in range(len(flist)):
    lb_table.insert(END, flist[i])
lb_table.grid(row=1, column=0)
lb_table.bind('<ButtonRelease-1>', get_list)
lb_table.bind('<KeyPress-Up>', get_list3)
lb_table.bind('<KeyPress-Down>', get_list4)


sb_field = Scrollbar(field_frame)
sb_field.grid(row=1, column=1, sticky=N+S)

lb_field = Listbox(field_frame, selectmode=EXTENDED, yscrollcommand=sb_field.set, width=80, height=28)
lb_field.grid(row=1, column=0)


sb_select = Scrollbar(selected_frame)
sb_select.grid(row=1, column=1, sticky=N+S)

lb_select = Listbox(selected_frame, selectmode=EXTENDED, yscrollcommand=sb_select.set, width=80, height=28)
lb_select.grid(row=1, column=0)
lb_select.bind('<ButtonRelease-1>', get_list2)



#listbox5 = Listbox(btmrt_frame, selectmode=EXTENDED, yscrollcommand=scrollbar4.set, width=80, height=14)
#listbox2.pack(side=BOTTOM, fill=BOTH)
#listbox5.grid(row=1, column=0)

sb_table.config(command=lb_table.yview)
sb_field.config(command=lb_field.yview)
sb_select.config(command=lb_select.yview)


mainloop()
#"""

#height	Number of lines (not pixels!)
#width	The width of the widget in characters (not pixels!)
#http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/listbox.html
#

"""
def add_item():
    #add the text in the Entry widget to the end of the listbox
    listbox1.insert(tk.END, enter1.get())
def delete_item():
    #delete a selected line from the listbox
    try:
        # get selected line index
        index = listbox1.curselection()[0]
        listbox1.delete(index)
    except IndexError:
        pass
 
def get_list(event):
    #function to read the listbox selection
    #and put the result in an entry widget
    # get selected line index
    index = listbox1.curselection()[0]
    # get the line's text
    seltext = listbox1.get(index)
    # delete previous text in enter1
    enter1.delete(0, 50)
    # now display the selected text
    enter1.insert(0, seltext)
def set_list(event):
    #insert an edited line from the entry widget
    #back into the listbox
    try:
        index = listbox1.curselection()[0]
        # delete old listbox line
        listbox1.delete(index)
    except IndexError:
        index = tk.END
    # insert edited item back into listbox1 at index
    listbox1.insert(index, enter1.get())
def sort_list():
   # function to sort listbox items case insensitive
    temp_list = list(listbox1.get(0, tk.END))
    temp_list.sort(key=str.lower)
    # delete contents of present listbox
    listbox1.delete(0, tk.END)
    # load listbox with sorted data
    for item in temp_list:
        listbox1.insert(tk.END, item)
def save_list():
    #save the current listbox contents to a file
    # get a list of listbox lines
    temp_list = list(listbox1.get(0, tk.END))
    # add a trailing newline char to each line
    temp_list = [chem + '\n' for chem in temp_list]
    # give the file a different name
    #fout = open("chem_data2.txt", "w")
    #fout.writelines(temp_list)
    #fout.close()
    print(temp_list)
 
# create the sample data file
str1 = [ethyl alcohol
ethanol
ethyl hydroxide
hydroxyethane
methyl hydroxymethane
ethoxy hydride
gin
bourbon
rum
schnaps
]
#fout = open("chem_data.txt", "w")
#fout.write(str1)
#fout.close()
 
# read the data file into a list
#fin = open("chem_data.txt", "r")
#chem_list = fin.readlines()
#fin.close()
# strip the trailing newline char
#chem_list = [chem.rstrip() for chem in chem_list]
chem_list = ['ethyl alcohol', 'ethanol', 'hydroxeyethane']

root = tk.Tk()
root.title("Listbox Operations")
# create the listbox (note that size is in characters)
listbox1 = tk.Listbox(root, width=50, height=6)
listbox1.grid(row=0, column=0)
 
# create a vertical scrollbar to the right of the listbox
yscroll = tk.Scrollbar(command=listbox1.yview, orient=tk.VERTICAL)
yscroll.grid(row=0, column=1, sticky=tk.N+tk.S)
listbox1.configure(yscrollcommand=yscroll.set)
 
# use entry widget to display/edit selection
enter1 = tk.Entry(root, width=50, bg='yellow')
enter1.insert(0, 'Click on an item in the listbox')
enter1.grid(row=1, column=0)
# pressing the return key will update edited line
enter1.bind('<Return>', set_list)
# or double click left mouse button to update line
enter1.bind('<Double-1>', set_list)
# button to sort listbox
button1 = tk.Button(root, text='Sort the listbox    ', command=sort_list)
button1.grid(row=2, column=0, sticky=tk.W)
# button to save the listbox's data lines to a file
button2 = tk.Button(root, text='Save lines to file', command=save_list)
button2.grid(row=3, column=0, sticky=tk.W)
# button to add a line to the listbox
button3 = tk.Button(root, text='Add entry text to listbox', command=add_item)
button3.grid(row=2, column=0, sticky=tk.E)
# button to delete a line from listbox
button4 = tk.Button(root, text='Delete selected line     ', command=delete_item)
button4.grid(row=3, column=0, sticky=tk.E)
# load the listbox with data
for item in chem_list:
    listbox1.insert(tk.END, item)
 
# left mouse click on a list item to display selection
listbox1.bind('<ButtonRelease-1>', get_list)
"""
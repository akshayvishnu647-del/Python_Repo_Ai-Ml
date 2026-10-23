import tkinter as tk
from tkinter import ttk

class CalculatorApp:
    def __init__(self,root):#__init__ to initailize object
        self.root=root
        self.root.title("Calculator")
        self.root.geometry("300x400")
        self.root.resizable(True,True)

        self.expression=""
        #create a string variable to display the texxt
        self.display_text=tk.StringVar()
        
        #create display frame
        display_frame=ttk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH,expand=True)

        #create the display label
        display_label=ttk.Label(
            display_frame,
            textvariable=self.display_text,
            font=("Arial",26),
            anchor="e",
            background="white",
            foreground="black",
            padding=6

        )
        display_label.pack(fill=tk.BOTH,expand=True)

        #create a button frame
        button_frame=ttk.Frame(self.root)
        button_frame.pack(fill=tk.BOTH,expand=True)

        self.create_buttons(button_frame)
    
    def create_buttons(self,frame):
        button=[
            ('7',1,0),('8',1,1),('9',1,2),('/',1,3),
            ('4',2,0),('5',2,1),('6',2,2),('*',2,3),
            ('1',3,0),('2',3,1),('3',3,2),('-',3,3),
            ('C',4,0),('0',4,1),('=',4,2),('+',4,3),
        ]


        for (text,row,col) in button:
            button=ttk.Button(frame,text=text,command=lambda t=text:self.on_button_click(t))
            button.grid(row=row,column=col,sticky="nsew",padx=2,pady=2)


        #configure rows and columns weight for resizing 
        for i in range(5):
            frame.rowconfigure(1,weight=1)
            frame.columnconfigure(1,weight=1)

        

    def on_button_click(self,button_text):
        if button_text == "C":
            self.expression=""
            
        elif button_text=="=":
            try:
                self.expression=str(eval(self.expression))

            except Exception as e:
                self.expression="Error"

        else:
            self.expression+=button_text

        self.display_text.set(self.expression)

if __name__=="__main__":
    root=tk.Tk()
    app=CalculatorApp(root)
    root.mainloop()




import tkinter as tk
from run_translation import translate
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Click me to say English! Once clicked, speak into the mic"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        output_string = ""
        for key, value in translate().items():
            output_string += f"{key}: {value}, \n"
        self.message = tk.Message(self.master, text=output_string)
        self.message.pack()

root = tk.Tk()
app = Application(master=root)
app.mainloop()

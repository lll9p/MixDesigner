#!/usr/bin/env python
#
#from numpy import arange, sin, pi
#
#import matplotlib
#matplotlib.use('TkAgg')
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
## implement the default mpl key bindings
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
    import Tkinter.ttk as ttk
else:
    import tkinter as tk
    import tkinter.ttk as ttk


class DisignerWindow():

    def __init__(self, master=None):
        self.master = master


class Application():

    class ModelSelector():

        def __init__(self, master=None):
            self.master = master
            self.header = []
            self.demands = []
            for i, text in enumerate(('N.O.', 'Min', 'x', 'Max')):
                self.header.append(
                    tk.Label(master=self.master, text=text).grid(row=0, column=i))
            self.header.append(tk.Button(master=self.master, text='ADD',
                                         command=self._add_demand).grid(row=0, column=len(self.header)))

        def get_design_demands(self):
            design_demand = {}
            for demand in self.demands:
                design_demand[demand[2].get()] = {'min': demand[
                    1].get(), 'max': demand[3].get()}
            return design_demand

        def _add_demand(self):
            demand_size = len(self.demands)
            demand = []
            demand.extend([
                tk.Label(master=self.master),
                tk.Entry(master=self.master),
                tk.Entry(master=self.master, textvariable=tk.StringVar(
                    self.master, ''.join(('x', str(demand_size + 1))))),
                tk.Entry(master=self.master),
                tk.Button(master=self.master, text='DEL',
                          command=lambda demand=demand: self._remove_demand(demand))
            ])
            for i, widgets in enumerate(demand):
                widgets.grid(
                    row=demand_size + 1, column=i)
            self.demands.append(demand)
            self._renew_NO()

        def _remove_demand(self, demand):
            for widget in demand:
                widget.destroy()
            self.demands.remove(demand)
            self._renew_NO()

        def _renew_NO(self):
            for i, demand in enumerate(self.demands):
                demand[0].config(text=str(i + 1))
                demand[0].update_idletasks()

    def __init__(self, master=None):
        self.master = master
        master.title('Mix designer')
        self.master.config(menu=self.create_menu())
        self.model_selector = self.ModelSelector(
            master=tk.Frame(master=self.master))
        self.model_selector.master.grid()

    def create_menu(self):
        menu_bar = tk.Menu(master=self.master)
        file_menu = tk.Menu(master=menu_bar, tearoff=0)
        file_menu.add_command(label='Open')
        file_menu.add_command(label='Save as')
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self._exit)

        menu_bar.add_cascade(label='File', menu=file_menu)

        design_menu = tk.Menu(master=menu_bar, tearoff=0)
        design_menu.add_command(label='Design')


        menu_bar.add_cascade(label='Design', menu=design_menu)
        self.menu_bar = menu_bar
        return self.menu_bar

    def _exit(self):
        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry('')
    app = Application(root)
    root.mainloop()

'''
root = tk.Tk()
root.wm_title("Embedding in TK")


f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)

a.plot(t, s)


# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=tk.BOTTOM)

tk.mainloop()
# If you put root.destroy() here, it will cause an error if
# the window is closed with the window manager.
'''


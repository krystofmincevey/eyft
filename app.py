import tkinter as tk
from tkinter import filedialog
import pandas as pd


class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Data Processing Application')
        self.create_widgets()

    def create_widgets(self):
        # Create the tabs
        self.tab_parent = tk.ttk.Notebook(self.root)

        self.tab1 = tk.ttk.Frame(self.tab_parent)
        self.tab2 = tk.ttk.Frame(self.tab_parent)
        self.tab3 = tk.ttk.Frame(self.tab_parent)
        self.tab4 = tk.ttk.Frame(self.tab_parent)
        self.tab5 = tk.ttk.Frame(self.tab_parent)

        self.tab_parent.add(self.tab1, text='Problem Selection')
        self.tab_parent.add(self.tab2, text='Data Upload')
        self.tab_parent.add(self.tab3, text='Data Processing')
        self.tab_parent.add(self.tab4, text='Feature Selection')
        self.tab_parent.add(self.tab5, text='Model Training')

        self.tab_parent.pack(expand=1, fill='both')

        # Create the problem selection widgets
        self.create_problem_selection_widgets()

        # Create the data upload widgets
        self.create_data_upload_widgets()

        # Create the data processing widgets
        self.create_data_processing_widgets()

        # Create the feature selection widgets
        self.create_feature_selection_widgets()

        # Create the model training widgets
        self.create_model_training_widgets()

    def create_problem_selection_widgets(self):
        pass  # TODO

    def create_data_upload_widgets(self):
        pass  # TODO

    def create_data_processing_widgets(self):
        pass  # TODO

    def create_feature_selection_widgets(self):
        pass  # TODO

    def create_model_training_widgets(self):
        pass  # TODO


root = tk.Tk()
app = DataProcessingApp(root)
root.mainloop()
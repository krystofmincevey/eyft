import csv
import tkinter as tk
import pandas as pd
from tkinter import ttk, filedialog
from pandastable import Table


class DataUploadTab(ttk.Frame):

    filepath_var: tk.StringVar
    table_frame: tk.Frame
    table: Table

    def __init__(self, parent, callback):
        super().__init__(parent)
        self.callback = callback
        self.data: pd.DataFrame = None
        self.create_widgets()

    def create_widgets(self):
        self.filepath_var = tk.StringVar()

        tk.Button(self, text='Upload File', command=self.upload_file).pack()
        tk.Label(self, textvariable=self.filepath_var).pack()

        self.table_frame = tk.Frame(self)
        self.table_frame.pack(fill='both', expand=True)

        # Add a button for saving changes
        tk.Button(self, text='Save Changes', command=self.save_changes).pack()

    def upload_file(self):

        def guess_delimiter(filepath_):
            with open(filepath_, 'r', encoding='utf-8') as file:
                dialect = csv.Sniffer().sniff(file.read(1024))
            return dialect.delimiter

        filetypes = [
            ('Excel Files', '*.xlsx;*.xls'),
            ('CSV Files', '*.csv'),
            ('Text Files', '*.txt')
        ]

        filepath = filedialog.askopenfilename(filetypes=filetypes)

        if filepath:
            self.filepath_var.set(filepath)
            extension = filepath.split('.')[-1]

            if extension in ['xlsx', 'xls']:
                data = pd.read_excel(filepath)
            elif extension == 'csv':
                data = pd.read_csv(filepath)
            elif extension == 'txt':
                delimiter = guess_delimiter(filepath)
                data = pd.read_csv(filepath, delimiter=delimiter)
            else:
                raise FileNotFoundError(
                    f'upload_file supports csv, xlsx, xls, and txt inputs '
                    f'not {extension}.'
                )

            self.data = data
            self.callback(data)  # Set the data in the main app

            # Clear the old table frame
            for widget in self.table_frame.winfo_children():
                widget.destroy()

            # Create a new table with the loaded data
            self.table = Table(self.table_frame, dataframe=data, showtoolbar=True, showstatusbar=True)
            self.table.show()

    def save_changes(self):
        self.data = self.table.model.df
        self.callback(self.data)

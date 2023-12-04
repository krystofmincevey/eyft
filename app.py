import tkinter as tk
from tkinter import ttk
from tabs.data_upload import DataUploadTab
from tabs.data_processing import DataProcessingTab
from tabs.feature_selection import FeatureSelectionTab

# from tabs.feature_selection import
# FeatureSelectionTab, ModelTrainingTab
# )


class MachineLearningApp:

    tab_parent: ttk.Notebook
    tab1: ttk.Frame
    tab2: DataUploadTab
    tab3: DataProcessingTab
    tab4: ttk.Frame
    tab5: ttk.Frame
    tab6: ttk.Frame
    tab7: ttk.Frame
    problem_type: str

    def __init__(self, root):
        self.root = root
        self.root.title('EY-ML')
        self.data = None  # Add this line
        self.create_widgets()

    def create_widgets(self):
        # Create the tabs
        self.tab_parent = ttk.Notebook(self.root)

        self.tab1 = ttk.Frame(self.tab_parent)
        self.tab2 = DataUploadTab(self.tab_parent, self.set_data)  # Pass the method to set data
        self.tab3 = DataProcessingTab(self.tab_parent, self.set_split_data)  # pass the set_split_data method
        self.tab4 = FeatureSelectionTab(self.tab_parent)
        self.tab5 = ttk.Frame(self.tab_parent)
        self.tab6 = ttk.Frame(self.tab_parent)
        self.tab7 = ttk.Frame(self.tab_parent)

        self.tab_parent.add(self.tab1, text='Problem Selection')
        self.tab_parent.pack(expand=1, fill='both')

        # Create the problem selection widgets
        self.create_problem_selection_widgets()

    def create_problem_selection_widgets(self):
        # Create a main frame in the tab to hold all elements
        main_frame = ttk.Frame(self.tab1)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Create a label to describe the choices the user has
        desc_label = ttk.Label(main_frame, text="Select the Problem Type:", font=('Arial', 14))
        desc_label.pack(side='top', pady=20)

        # Create a frame to hold the buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side='top')

        # Place the buttons in the button frame, in a grid
        time_series_button = ttk.Button(
            button_frame, text='Time Series',
            command=lambda: self.problem_selected('timeseries')
        )
        time_series_button.grid(row=0, column=0, padx=20, pady=20)

        cross_section_button = ttk.Button(
            button_frame, text='Cross Section',
            command=lambda: self.problem_selected('cross_section')
        )
        cross_section_button.grid(row=0, column=1, padx=20, pady=20)

    def problem_selected(self, problem_type):
        self.problem_type = problem_type
        # FOR NOW IF ELSE DOES NOT CHANGE ANYTHING - but in the future it will
        if problem_type == 'timeseries':
            # handle time series choice
            self.tab_parent.add(self.tab2, text='Data Upload')
            self.tab_parent.add(self.tab3, text='Data Processing')
            self.tab_parent.add(self.tab4, text='Feature Selection')
            self.tab_parent.add(self.tab5, text='GLM')
            self.tab_parent.add(self.tab6, text='Decision Tree')
            self.tab_parent.add(self.tab7, text='XGBoost')
            self.tab_parent.hide(0)  # Hide the problem selection tab
        elif problem_type == 'cross_section':
            self.tab_parent.add(self.tab2, text='Data Upload')
            self.tab_parent.add(self.tab3, text='Data Processing')
            self.tab_parent.add(self.tab4, text='Feature Selection')
            self.tab_parent.add(self.tab5, text='GLM')
            self.tab_parent.add(self.tab6, text='Decision Tree')
            self.tab_parent.add(self.tab7, text='XGBoost')
            self.tab_parent.hide(0)  # Hide the problem selection tab

    def set_data(self, data):
        self.data = data
        self.tab3.update_widgets(
            self.data, self.problem_type
        )  # Update the data processing tab

    def set_split_data(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.tab4.update_feature_selection_widgets(train_data, test_data)


if __name__ == "__main__":
    root = tk.Tk()
    app = MachineLearningApp(root)
    root.mainloop()

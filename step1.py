print('''
------------------   ^_^   ------------------
Processing Tool for Step 1.
This program was written by Dawei Wen (温大尉)(*^_^*).
If you have any question, please contact:
E-mail: ontaii@163.com.
Google Scholar: https://scholar.google.com/citations?hl=ja&user=U13L9sEAAAAJ
Research Gate:  https://www.researchgate.net/profile/Dawei-Wen-2
------------------   ^_^   ------------------
''')

import pandas as pd

# Integration function
def Integrate(x, y):
    S = 0
    Counts = len(x)
    for i in range(1, Counts):
        S += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2
    return S

# Define wavelength ranges
Wavelength_Groups = {
    "600to675": (lambda x: 600 <= x < 675),
    "675to700": (lambda x: 675 <= x < 700),
    "700to750": (lambda x: 700 <= x < 750),
    "750to950": (lambda x: 750 <= x < 950),
}

#Load the Excel File
Input_Spectra = input('Enter the file name of the spectra with extension (e.g., aa.xlsx):\n')
Spectra = pd.read_excel(Input_Spectra)

#Identify and sort pressure columns
pressure_columns = [col for col in Spectra.columns if col != 'Wavelength']
pressure_values = [float(col) for col in pressure_columns]

# Sort by pressure
pressure_pairs = sorted(zip(pressure_values, pressure_columns))
pressure_values = [p for p, _ in pressure_pairs]
pressure_columns = [col for _, col in pressure_pairs]

#Group data by wavelength range
Grouped_Data = {
    group_name: {
        "Wavelength": [],
        "Pressures": {p: [] for p in pressure_values}
    }
    for group_name in Wavelength_Groups
}

for idx, row in Spectra.iterrows():
    wl = row['Wavelength']
    for group_name, condition in Wavelength_Groups.items():
        if condition(wl):
            Grouped_Data[group_name]["Wavelength"].append(wl)
            for p_val, col in zip(pressure_values, pressure_columns):
                Grouped_Data[group_name]["Pressures"][p_val].append(row[col])
            break  # stop after finding the first matching group

#Integration under each pressure
Area_Data = {group: [] for group in Wavelength_Groups}

for p in pressure_values:
    for group in Wavelength_Groups:
        wl = Grouped_Data[group]['Wavelength']
        intensity = Grouped_Data[group]['Pressures'][p]
        Area_Data[group].append(Integrate(wl, intensity))

#Create DataFrame and Normalize
Matrix_for_ML = pd.DataFrame(Area_Data)
Matrix_for_ML['Pressure (GPa)'] = pressure_values

# Normalize each group (%)
denominator = Matrix_for_ML[list(Wavelength_Groups.keys())].sum(axis=1)
for col in Wavelength_Groups:
    Matrix_for_ML[f'{col} (%)'] = Matrix_for_ML[col] / denominator * 100

#Add ratio column
Matrix_for_ML['LIR'] = Matrix_for_ML['675to700 (%)'] / Matrix_for_ML['700to750 (%)']

#Save the output
Output_Matrix_Name = input('Enter the file name of the output matrix (e.g., Output.xlsx):\n')
Matrix_for_ML.to_excel(Output_Matrix_Name, index=False)

print('Finished.')
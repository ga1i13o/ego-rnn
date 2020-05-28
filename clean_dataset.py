import os
TO_DELETE = ".DS_Store"

def clean_data(directory):
    print(directory)
    elements = os.listdir(directory)
    if TO_DELETE in elements:
        print("removing ..  "+os.path.join(directory, elements[elements.index(TO_DELETE)]))
        os.remove(os.path.join(directory, elements[elements.index(TO_DELETE)]))
    for el in elements:
        d = os.path.join(directory, el)
        if os.path.isdir(d):
            clean_data(d)

root = "/home/utente/Scaricati/program/ML_DL/FPAR/GTEA61"
clean_data(root)
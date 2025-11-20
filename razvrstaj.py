import os
import shutil

def organizuj_slike_i_maske(ulazni_folder):
    # kreiranje novih foldera ako ne postoje
    folder_slike = os.path.join(ulazni_folder, "slike")
    folder_maske = os.path.join(ulazni_folder, "maske")

    os.makedirs(folder_slike, exist_ok=True)
    os.makedirs(folder_maske, exist_ok=True)

    # iteriranje kroz sve fajlove u ulaznom folderu
    for fajl in os.listdir(ulazni_folder):
        if fajl.endswith(".tif"): 
            putanja_do_fajla = os.path.join(ulazni_folder, fajl)

            if "slika" in fajl.lower(): 
                shutil.move(putanja_do_fajla, os.path.join(folder_slike, fajl))
                print(f"Premesteno '{fajl}' u '{folder_slike}'")
            elif "maska" in fajl.lower():
                shutil.move(putanja_do_fajla, os.path.join(folder_maske, fajl))
                print(f"Premesteno '{fajl}' u '{folder_maske}'")

# putanja
ulazni_folder_za_obradu = r"C:\Users\Korisnik\OneDrive\Desktop\N"
organizuj_slike_i_maske(ulazni_folder_za_obradu)
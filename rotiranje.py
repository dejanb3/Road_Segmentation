from PIL import Image
import os
import re

def transformisi_slike_iz_podfoldera(glavni_ulazni_folder=r"C:\Users\Korisnik\OneDrive\Desktop\nale6"):
    # ulazni folderi
    ulazni_folder_slike = os.path.join(glavni_ulazni_folder, r"C:\Users\Korisnik\OneDrive\Desktop\nale6\slike")
    ulazni_folder_maske = os.path.join(glavni_ulazni_folder, r"C:\Users\Korisnik\OneDrive\Desktop\nale6\maske")

    # izlazni folderi
    glavni_izlazni_folder_transformed = os.path.join(glavni_ulazni_folder, "transformed_output")
    os.makedirs(glavni_izlazni_folder_transformed, exist_ok=True)
    izlazni_folder_slike = os.path.join(glavni_izlazni_folder_transformed, "slike")
    izlazni_folder_maske = os.path.join(glavni_izlazni_folder_transformed, "maske")

    os.makedirs(izlazni_folder_slike, exist_ok=True)
    os.makedirs(izlazni_folder_maske, exist_ok=True)

    # rotacia
    uglovi_rotacije = [90, 180, 270]

    # vertikalne rotacija
    ofseti_v = {
        90: 10000,
        180: 20000,
        270: 30000
    }

    # horizontalna rotacija
    ofseti_h = {
        90: 40000,
        180: 50000,
        270: 60000
    }

    regex_broja = re.compile(r'(\d+)\.tif$', re.IGNORECASE)
    lista_ulaznih_foldera = [ulazni_folder_slike, ulazni_folder_maske]

    for ulazni_subfolder in lista_ulaznih_foldera:
        if not os.path.isdir(ulazni_subfolder):
            print(f"Upozorenje: Ulazni folder '{ulazni_subfolder}' ne postoji. Preskacem.")
            continue

        for ime_fajla in os.listdir(ulazni_subfolder):
            if ime_fajla.lower().endswith(".tif"):
                putanja_do_fajla = os.path.join(ulazni_subfolder, ime_fajla)

                target_output_folder = None
                if "slika" in ime_fajla.lower():
                    target_output_folder = izlazni_folder_slike
                elif "maska" in ime_fajla.lower():
                    target_output_folder = izlazni_folder_maske
                else:
                    print(f"Upozorenje: Nepoznat tip fajla '{ime_fajla}' u '{ulazni_subfolder}'. Mora sadrzati 'slike' ili 'maska'. Preskacem.")
                    continue

                match = regex_broja.search(ime_fajla)
                originalni_broj = None
                originalni_broj_str = ""
                if match:
                    originalni_broj_str = match.group(1)
                    originalni_broj = int(originalni_broj_str)
                else:
                    print(f"Upozorenje: Nije pronadjen broj u nazivu fajla '{ime_fajla}'. Preskacem obradu ovog fajla.")
                    continue

                try:
                    with Image.open(putanja_do_fajla) as img:
                        print(f"Obradjujem sliku: {ime_fajla} iz '{os.path.basename(ulazni_subfolder)}' (originalni broj: {originalni_broj})")

                        for ugao in uglovi_rotacije:
                            rotirana_slika = img.rotate(ugao, expand=True)
                            ofset = ofseti_v[ugao]
                            novi_broj = originalni_broj + ofset

                            prefiks_naziva = ime_fajla[:ime_fajla.find(originalni_broj_str)]
                            novi_naziv = f"{prefiks_naziva}{novi_broj}.tif"

                            putanja_za_cuvanje = os.path.join(target_output_folder, novi_naziv)
                            rotirana_slika.save(putanja_za_cuvanje, compression="tiff_lzw")
                            print(f"  Sacuvano (VERT rotacija {ugao}° u {os.path.basename(target_output_folder)}): {novi_naziv}")

                        for ugao in uglovi_rotacije:
                            rotirana_slika = img.rotate(ugao, expand=True)
                            mirror_rotirana_slika = rotirana_slika.transpose(Image.FLIP_LEFT_RIGHT)

                            ofset = ofseti_h[ugao]
                            novi_broj = originalni_broj + ofset

                            prefiks_naziva = ime_fajla[:ime_fajla.find(originalni_broj_str)]
                            novi_naziv = f"{prefiks_naziva}{novi_broj}.tif"

                            putanja_za_cuvanje = os.path.join(target_output_folder, novi_naziv)
                            mirror_rotirana_slika.save(putanja_za_cuvanje, compression="tiff_lzw")
                            print(f"  Sacuvano (HORIZ mirror-rotacija {ugao}° u {os.path.basename(target_output_folder)}): {novi_naziv}")

                except Exception as e:
                    print(f"Greska pri obradi fajla {ime_fajla}: {e}")

if __name__ == "__main__":
    transformisi_slike_iz_podfoldera()
    print("\nProces transformacije iz podfoldera zavrsen!")
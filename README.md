# Segmentacija Puteva Pomoću U-Net Arhitekture

Ovaj projekat implementira U-Net model za semantičku segmentaciju puteva na satelitskim snimcima koristeći PyTorch. Projekat obuhvata ceo proces, od pripreme podataka i augmentacije, preko treniranja i validacije, do finalne evaluacije i post-obrade rezultata.

---

### Opis Projekta

Celokupan proces se može podeliti u četiri glavne faze:

1.  **Priprema podataka:** Uključuje sečenje velikih satelitskih snimaka, augmentaciju podataka (rotiranje i preslikavanje) i organizaciju fajlova.
2.  **Treniranje modela:** Definicija U-Net arhitekture, učitavanje podataka i treniranje modela na pripremljenom skupu, uz korišćenje validacionog skupa za sprečavanje overfittinga.
3.  **Post-obrada:** Primena heurističkih algoritama (morfoloških transformacija) na sirove predikcije modela kako bi se uklonio šum i popunili prekidi na putnoj mreži.
4.  **Evaluacija i predikcija:** Merenje performansi finalnog sistema (model + post-obrada) na testnom skupu i primena modela na velikoj slici radi vizuelnog rezultata.

---

### Korišćeni fajlovi

*   **`road.py`**: Glavna skripta projekta. Sadrži kompletnu implementaciju U-Net modela, klase za učitavanje podataka, funkcije za treniranje, validaciju, finalnu evaluaciju i predikciju na velikim slikama pomoću tehnike kliznog prozora. Takođe uključuje i funkcije za post-obradu.
*   **`rotiranje.py`**: Skripta za augmentaciju podataka. Povećava veličinu trening skupa tako što kreira nove verzije postojećih slika i maski njihovim rotiranjem (za 90, 180, 270 stepeni) i horizontalnim/vertikalnim preslikavanjem.
*   **`razvrstaj.py`**: Pomoćna skripta za organizaciju fajlova. Premešta fajlove koji u imenu sadrže "slika" u podfolder `images`, a one koji sadrže "maska" u podfolder `maske`.


---

### Kako pokrenuti

Glavni kod za treniranje i evaluaciju (`road.py`) je predviđen za izvršavanje na platformi Google Colab.

1.  **Postaviti podatke** (slike i maske) na Google Drive u sledeću strukturu:
    ```
    /content/drive/MyDrive/
    └── data/
        ├── train/
        │   ├── images/
        │   └── maska/
        ├── val/
        │   ├── images/
        │   └── maska/
        └── test/
            ├── images/
            └── maska/
    ```

2.  **Otvoriti `road.py`** u Google Colab okruženju.

3.  **Pokrenuti željenu operaciju** (npr. `run_training()`, `run_final_evaluation()` ili `example_usage()`) otkomentarisanjem odgovarajuće linije u `if __name__ == "__main__"` bloku na dnu skripte.


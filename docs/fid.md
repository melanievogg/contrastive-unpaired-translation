# FID für DOCTR/CUT

Installation in derselben Python-Umgebung wie CUT:

```bash
python3 -m pip install pytorch-fid==0.3.0
```

Vom Projektverzeichnis aus:

```bash
# SELFF -> SD
python3 calculate_fid.py --real ../doctr_cut_2d_matched/testB --fake ./results/doctr_SELFF_to_SD_nl2/test_latest/images/fake_B --direction AtoB --output-csv fid_results.csv --label matched_nlayers2

# SD -> SELFF (Ergebnisordner ggf. an tatsächlichen Experimentnamen anpassen)
python3 calculate_fid.py --real ../doctr_cut_2d_matched/testA --fake ./results/doctr_SD_to_SELFF_nl2/test_latest/images/fake_A --direction BtoA --output-csv fid_results.csv --label matched_reverse_nlayers2
```

`--batch-size 32` ist der Standard. `--device cpu` oder `--device cuda:0`
überschreibt die automatische Geräteauswahl. Beim ersten Lauf lädt pytorch-fid
seine FID-Inception-Gewichte herunter; dafür ist Internetzugriff erforderlich.
Das Skript verändert keine Bilder. Die CSV wird angehängt und muss den passenden
Header haben. Parallele Schreibzugriffe auf dieselbe CSV vermeiden.

Die Referenz muss der vollständige originale Ziel-Testordner `testB` (AtoB)
oder `testA` (BtoA) sein. Nur generierte Testbilder in `fake_B` bzw. `fake_A`
übergeben. Verzeichnisse namens train/trainA/trainB/training werden abgewiesen,
auch bei aufgelösten Dateisymlinks. Die Herkunft beliebig umbenannter oder
kopierter Bilder lässt sich nicht automatisch feststellen und muss geprüft werden.
Rekursive Suche: PNG, JPG, JPEG, TIF, TIFF, unabhängig von Groß-/Kleinschreibung.
Keine Paarzuordnung, keine Kürzung bei unterschiedlichen Bildzahlen.

Die [pytorch-fid-Implementierung](https://github.com/mseitzer/pytorch-fid/tree/v0.3.0)
liest jedes Bild explizit mit Pillow `convert('RGB')` ein. Bei 8-Bit-Graustufen
werden die Kanäle repliziert. Außer der standardmäßigen Umwandlung zu Float / 255
vor Inception gibt es keine eigene Intensitätsverarbeitung. Alpha wird bei RGB-
Konvertierung verworfen, Paletten werden auf RGB aufgelöst. Hochbitige oder
Float-Graustufen werden abgewiesen, um stilles Clipping zu vermeiden;
mehrseitige TIFFs müssen zuvor als einzelne B-Scans bereitgestellt werden.

Originalgrößen bleiben bis zum Inception-Aufruf erhalten. Unterschiedliche Größen
werden separat gebatcht; sämtliche 2048-dimensionalen Features werden anschließend
zu genau einer Verteilung pro Domain zusammengeführt. Nur die standardmäßige
Inception-Verarbeitung führt intern Resize auf 299 x 299 und Skalierung auf [-1, 1]
aus. Es gibt kein eigenes Crop, Padding, Resize, Histogram Matching oder
bildabhängiges Normalisieren. Mindestens zwei Bilder pro Domain sind nötig.

Sortierte Dateien, feste Seeds und deterministische Torch-Operationen dienen der
Reproduzierbarkeit. Paketversionen werden ausgegeben; für Modellvergleiche dieselbe
Umgebung, Gewichte, Hardware, Batchgröße, Implementierung und exakt denselben
Testsplit verwenden und die Konsolenausgabe archivieren. Plattformübergreifend
ist keine bitweise Gleichheit garantiert.

Kleinerer FID bedeutet, dass die Feature-Verteilung der generierten Bilder näher
an der echten Ziel-Domain liegt. FID prüft **nicht direkt die Erhaltung der Anatomie**
von real_A in fake_B und darf nicht als alleinige Metrik anatomischer Konsistenz
interpretiert werden. Inception wurde für natürliche Bilder entwickelt. Die
Stichprobengröße beeinflusst FID; bei etwa 1280 Bildern und 2048 Features ist die
empirische Kovarianz rangdefizient. Deshalb nur kontrollierte Vergleiche mit
denselben Testmengen durchführen, keine absoluten klinischen Aussagen ableiten.

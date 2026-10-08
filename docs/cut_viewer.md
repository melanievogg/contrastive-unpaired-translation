# CUT Visual QC Viewer

Aufruf aus dem Repository (Python 3, keine Zusatzpakete):

```bash
python build_cut_viewer.py \
  --results-dir checkpoints/doctr_SELFF_to_SD_fixed_nl2/web/images \
  --output-dir checkpoints/doctr_SELFF_to_SD_fixed_nl2/viewer \
  --title "CUT SELFF → SD fixed n_layers=2 · Trainingsvorschau nach Epoche"

python build_cut_viewer.py \
  --results-dir results/doctr_SELFF_to_SD_fixed_nl2/test_latest/images \
  --output-dir results/doctr_SELFF_to_SD_fixed_nl2/test_latest/viewer \
  --title "CUT SELFF → SD fixed n_layers=2 · Test nach Epoche"

```

`index.html` direkt im Browser öffnen. Alternativ:

```bash
python -m http.server 8000 --directory checkpoints/doctr_SELFF_to_SD_fixed_nl2/viewer

python -m http.server 8000 --directory results/doctr_SELFF_to_SD_fixed_nl2/test_latest/viewer 

```

Dann http://localhost:8000 öffnen. Der Ausgabeordner ist portabel: `assets/` enthält unveränderte Byte-Kopien der Bilder. Dafür wird zusätzlicher Speicherplatz benötigt. Die Quellen werden nicht verändert. Eingabe- und Ausgabeordner dürfen nicht ineinander liegen.

Unterstützt werden `<basename>_<role>.png` und `<role>/<basename>.png` für `fake_B`, `real_A`, `real_B`, `idt_B`; außerdem JPG/JPEG, WebP, BMP und GIF. Rollen-Unterordner werden automatisch erkannt. `--recursive` durchsucht weitere Unterordner. Mehrdeutige Kombinationen aus Basename und Rolle führen zu einer Fehlermeldung, statt eine Datei willkürlich auszuwählen. Nicht passende Dateien werden ignoriert. Auch Gruppen mit nur einem Bild bleiben sichtbar.

Die Reihenfolge ist immer `fake_B | real_A | real_B | idt_B`. Vollständig erkannte Namen wie `subject018_OS_bscan_0043` werden numerisch nach Subject, Eye und B-Scan sortiert; andere Namen folgen natürlich alphanumerisch. Unbekannte Metadaten bleiben leer. Bei gleichem Sortierschlüssel entscheidet der vollständige Basename deterministisch.

Navigation: Previous/Next, Pfeiltasten, First/Last oder Fallnummer. Bild anklicken für Zoom, Fit für Übersicht, 100% für Originalpixel (bei großen Bildern scrollen). Escape, Schließen oder Klick außerhalb schließt den Zoom. CSS skaliert lediglich die Anzeige; es werden keine Bildwerte verändert.

Pro Fall gibt es Overall QC, Anatomy preservation, Target-domain appearance und Identity preservation. Nochmals auf die gewählte Bewertung klicken hebt sie auf. Bewertungen werden unter einem aus dem absoluten Eingabepfad abgeleiteten Schlüssel in localStorage gespeichert. Gleiche Eingabepfade teilen Bewertungen innerhalb desselben Browser-Ursprungs; verschiedene Datensätze sind getrennt. Bei Änderungen der Bilder unter gleichen Namen alte Bewertungen prüfen. Browser, Port, file:// gegenüber HTTP und Browserprofile können getrennte Speicherbereiche verwenden. Bei blockiertem Speicher erscheint ein Hinweis; die Bewertungen bleiben bis zum Schließen im Arbeitsspeicher. JSON regelmäßig über „Export QC ratings“ sichern. Der Export enthält alle Fälle, auch unbewertete, fehlende Rollen und Zeitstempel. Ein Import ist nicht implementiert.

`viewer_manifest.csv` enthält absolute Originalpfade, Vollständigkeit und erkannte Metadaten. `missing_files.csv` enthält eine Zeile je fehlender Rolle; erwartete Dateinamen sind Formatvorschläge, da die tatsächliche Dateiendung unbekannt ist. Beide Reports erhalten auch ohne Fälle ihre Kopfzeile. Wiederholtes Erzeugen aktualisiert HTML und Reports; frühere, nicht mehr referenzierte Asset-Kopien werden nicht gelöscht.

**Unpaired-Interpretation:** real_B ist keine Ground Truth für fake_B. Anatomy preservation vergleicht real_A → fake_B, Domain appearance beurteilt nur stilistische Plausibilität zur Ziel-Domain, Identity preservation vergleicht real_B → idt_B. Fehlende Identity-Ausgaben bleiben als „missing“ sichtbar; der Viewer erzeugt sie nicht. Es werden keine quantitativen Metriken berechnet.

Die aktuell verwendeten Checkpoint-Bilder heißen `epoch001_fake_B.png` usw. Eine Viewer-Gruppe entspricht deshalb einer gespeicherten Trainingsvorschau pro Epoche, nicht einem identifizierten Testfall. Der vollständige Epochenname wird angezeigt; Subject/Eye/B-Scan werden nicht erfunden.

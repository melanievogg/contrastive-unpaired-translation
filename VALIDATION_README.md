# CUT Validation und finale Evaluation

## Geänderte Dateien

`train.py` sammelt bildgewichtete Trainingsmittel und ruft die Epochenauswertung auf. `util/cut_evaluation.py` enthält Validation, FID, Cache, CSV-Persistenz und Diagramme. `evaluate_cut.py` wertet den besten Generator separat aus. `options/train_options.py` ergänzt CLI-Optionen. `data/unaligned_dataset.py` prüft Splits und entfernt den Test-zu-Validation-Fallback. `models/base_model.py` vermeidet CUDA-DataParallel bei explizitem CPU-Betrieb auf einem Rechner mit sichtbaren GPUs. `models/cut_model.py` unterdrückt inaktives NCE_Y-Logging und verwendet den F-Optimizer nur bei aktivem, trainierbarem MLP. Die mathematischen Loss-Funktionen und die Architektur bleiben unverändert. `requirements.txt` ergänzt Matplotlib, NumPy, SciPy und pytorch-fid. `tests/test_cut_evaluation.py` enthält synthetische CPU-Tests.

## Installation und Splits

```bash
cd /home/vogg/contrastive-unpaired-translation
python -m pip install -r requirements.txt
```

Erforderliche Struktur:

```text
datasets/my_dataset/
  trainA/  trainB/
  valA/    valB/
  testA/   testB/
```

A und B sind ungepaart und dürfen unterschiedlich viele Bilder enthalten. Fehlende oder leere Domänen erzeugen Fehler. Training lädt nur train, Validation nur val. Test wird ausschließlich im separaten Evaluationsskript geöffnet. Bei medizinischen Volumendaten müssen die Splits bereits auf Subjekt-/Volumenebene getrennt sein; der Bildloader erstellt diese Trennung nicht.

## Training

```bash
python train.py \
  --dataroot /home/vogg/doctr_cut_2d_fixed \
  --name my_cut --model cut --CUT_mode CUT --direction AtoB \
  --gpu_ids 0 --batch_size 1 --load_size 286 --crop_size 256 \
  --n_epochs 100 --n_epochs_decay 100 \
  --enable_validation --enable_fid --fid_freq 5 \
  --eval_size 256 --eval_seed 0 --num_val_images 5 \
  --display_id -1 --no_html
```

Der Dataroot muss die sechs oben gezeigten Ordner enthalten. Für CPU `--gpu_ids -1` verwenden. FastCUT mit `--CUT_mode FastCUT` ist ebenfalls unterstützt. `--enable_validation` und `--enable_fid` sind standardmäßig aus, damit alte Aufrufe mit ausschließlich Trainingsdaten weiter funktionieren. Beide Optionen unterstützen explizites `false`. FID kann auch ohne Validation-Loss-Diagnostik aktiviert werden; valA und valB sind dann dennoch erforderlich.

FID wird standardmäßig alle fünf Epochen sowie in der letzten Epoche berechnet. `--fid_at_end false` deaktiviert den zusätzlichen letzten Messpunkt. Training liest niemals Testdaten. Keine Teilmenge: FID verwendet immer die vollständigen Splits, unabhängig von `--max_dataset_size` des Trainings. Jede Quelle und jedes reale Ziel werden für FID genau einmal gezählt, auch bei unterschiedlichen Domänengrößen. `BtoA` verwendet entsprechend G(B) gegen A.

## Validation und Losses

Nach jeder Epoche werden bei aktivierter Validation die vorhandenen `forward`, `compute_D_loss` und `compute_G_loss` aufgerufen. Es gibt kein `backward` und keinen Optimizer-Schritt. G, D und F wechseln vorübergehend in den Evaluationsmodus. Die Trainingskonfiguration bleibt erhalten, damit Identity-NCE verfügbar ist. Die datenabhängige F-Initialisierung erfolgt regulär im ersten Trainingsbatch vor der ersten Validation. Alle Berechnungen laufen unter `torch.no_grad()`; PatchNCE funktioniert dabei ohne mathematische Anpassung.

Validation nutzt den bestehenden UnalignedDataset mit `serial_batches=True`, `no_flip=True`, `batch_size=1`, null Worker und festem quadratischem Resize auf `--eval_size` (Vielfaches von vier). Keine zufälligen Crops oder Augmentierungen. Deterministisches Indexieren der beiden Domänen bedeutet keine Paarannahme. Für Loss-Diagnostik verwendet der Loader wie bisher die größere Domänenlänge und wiederholt ggf. Bilder der kleineren Domäne.

PatchNCE bekommt vorübergehend Batchgröße eins. Patch-Sampling verwendet `--eval_seed`; Python-, NumPy-, CPU-/CUDA-Zufallszustände, cuDNN-Einstellungen und individuelle Netzwerkmodi werden danach wiederhergestellt. FastCUT-Flip-Augmentierung ist während der Auswertung aus. Dies macht die Evaluation reproduzierbar, garantiert jedoch keine bitweise Reproduzierbarkeit des gesamten ursprünglichen Trainings über unterschiedliche Hardware oder PyTorch-Versionen.

CSV-Spalten: `epoch,G_GAN,D_real,D_fake,D_total,NCE,G_total`; bei aktivem Identity-NCE zusätzlich `NCE_Y` vor `G_total`. `D_total` ist der bereits berechnete `loss_D`, `G_total` der bereits berechnete `loss_G`. Originale GAN-/NCE-Gewichtungen bleiben enthalten, insbesondere die Mittelung von NCE und NCE_Y. Mittelwerte sind nach tatsächlich verarbeiteten Bildern gewichtet. Die ursprüngliche Trainings-Loader-Logik mit `drop_last=True` bleibt erhalten; ein unvollständiger Trainingsbatch wird somit weiterhin verworfen. Der Logger unterstützt unterschiedliche Batchgrößen. Validation-Losses dienen der Diagnose und wählen keine Checkpoints aus.

## FID und bestes Modell

Verwendet wird [pytorch-fid 0.3.0](https://github.com/mseitzer/pytorch-fid) mit den offiziellen FID-Inception-Gewichten und 2048 Features. Beim ersten Aufruf müssen diese Gewichte verfügbar sein oder heruntergeladen werden können. Input und echte Zielbilder verwenden denselben festen Resize. CUT-Tensoren werden über `clamp(-1,1)`, `(x+1)/2` in RGB-Fließkommatensoren [0,1] überführt; ein Kanal wird dreimal repliziert. Inception übernimmt seine Standardvorverarbeitung. FID wird aus Fließkommatensoren berechnet, nicht aus den quantisierten PNG-Vorschauen. Diese Definition konsistent beibehalten und nicht direkt mit FID anderer Vorverarbeitungen vergleichen.

FID vergleicht unabhängige Bildverteilungen; es ist kein pixelweiser Vergleich ungepaarter Bilder. Mindestens zwei Bilder pro Domäne sind erforderlich. Bei kleinen Stichproben, insbesondere unter 2048 Bildern, sind Kovarianzschätzung und FID instabil und verzerrt. Bei OCT/medizinischen Bildern sind ImageNet-Inception-Features nur eingeschränkt aussagekräftig; zusätzlich visuell und fachlich prüfen.

Reale Feature-Statistiken liegen unter `metrics/fid_cache/`. Der Cache-Schlüssel berücksichtigt vollständige Zielpfade, Dateigröße, Änderungszeit, Auflösung, Richtung und pytorch-fid-Version. Bei Änderungen am Datensatz den Cache löschen, falls Dateimetadaten bewusst unverändert gehalten wurden. Generierte Features werden jedes Mal neu berechnet; Inception wird über FID-Epochen wiederverwendet. GPU-Batches sind eins; CPU-Speicher enthält Feature-Matrizen und die 2048²-Kovarianzen. Die vollständige FID-Auswertung kann bei großen Datensätzen viel Zeit benötigen.

Nur ein niedrigerer Validation-FID speichert `best_net_G.pth`, `best_net_D.pth`, `best_net_F.pth` und `metrics/best_model.json`. `latest`- und nummerierte Checkpoints bleiben erhalten. Bei Gleichstand bleibt das bisherige beste Modell. Bei deaktiviertem FID wird kein best-Checkpoint ausgewählt.

## Finale Evaluation

```bash
python evaluate_cut.py \
  --dataroot /home/vogg/doctr_cut_2d_fixed \
  --name my_cut --model cut --CUT_mode CUT --direction AtoB \
  --gpu_ids 0 --epoch best --eval_size 256 --eval_seed 0
```

Netzwerkoptionen (`netG`, `ngf`, Normalisierung, Eingabe-/Ausgabekanäle usw.) müssen denen des Trainings entsprechen. Das Skript erzwingt `best`, prüft alle Splits und lädt im Inferenzmodus ausschließlich `best_net_G.pth`. Es berechnet Training-, Validation- und Test-FID mit demselben Generator. `--num_test` begrenzt diese finale FID nicht. Generierte Testbilder werden vollständig unter `test_images/` gespeichert. Ergebnisse: `metrics/test_metrics.json`. Die finale Evaluation verändert die best-Auswahl nicht.

## Ausgabedateien

Alles unter `checkpoints/<name>/`:

- `metrics/train_losses.csv`, `val_losses.csv`: Epochendiagnostik.
- `metrics/fid_scores.csv`: nur tatsächlich gemessene FID-Epochen.
- `metrics/best_model.json`: best_epoch und best_val_fid.
- `metrics/test_metrics.json`: finale drei FIDs und Checkpoint.
- `plots/`: Generator-, Discriminator-, GAN-, NCE-, optional Identity-NCE- und Real/Fake-Losses, nach jeder Epoche aktualisiert.
- `plots/fid_train_val.{png,pdf}`: echte Messpunkte und best-Marker.
- `plots/fid_final_comparison.{png,pdf}`: finaler Vergleich vom best-Checkpoint.
- `validation_images/epoch_<N>/`: erste feste `--num_val_images` Quellen, Input | Generated, bei jeder FID-Auswertung.

Diagramme verwenden Agg ohne Desktop, konsistente Train-/Val-Farben, PNG mit 300 dpi und PDF. Vorschauen sind keine FID-Teilmengen.

## Fortsetzen

```bash
python train.py --dataroot /home/vogg/doctr_cut_2d_fixed \
  --name my_cut --model cut --gpu_ids 0 \
  --continue_train --epoch latest --epoch_count 76 \
  --n_epochs 100 --n_epochs_decay 100 \
  --enable_validation --enable_fid --fid_freq 5 \
  --eval_size 256 --display_id -1 --no_html
```

`--epoch_count` muss zur nächsten Epoche des geladenen Checkpoints passen. Die bestehende CUT-Wiederaufnahme lädt Netzwerkgewichte; sie speichert ursprünglich keine Adam-/Scheduler-/RNG-Zustände und bietet daher keine bitgenaue Wiederaufnahme. Diese ursprüngliche Semantik bleibt erhalten. Historische CSV-Zeilen bleiben erhalten; eine erneut ausgeführte Epoche ersetzt nur ihren eigenen Eintrag. Bestwerte werden aus JSON wiederhergestellt. Ohne `--continue_train` verweigert der Logger eine Wiederverwendung vorhandener Metriken, damit ein neues Training keine historischen Kurven mit neuen Gewichten vermischt. Bei geänderter Loss-Konfiguration einen neuen Experimentnamen verwenden.

## Tests und Umgebungsprüfung

```bash
python -m pytest tests/test_cut_evaluation.py -q
```

Der Integrationstest verwendet echte Inception-Gewichte (ggf. Download). Andere FID-Tests ersetzen nur die teure Distanzberechnung bzw. den Evaluator und prüfen Splitzählung, Cache, Intervalle und Auswahl separat. Tests prüfen CPU-CUT und FastCUT, Identity-NCE, deaktiviertes NCE, unveränderte Gewichte/Optimizer/RNG, Checkpoint-Laden, CSV-Deduplizierung, Wiederaufnahme und PNG/PDF-Ausgaben.

In der bereitgestellten Standardumgebung waren dill, Matplotlib, pytest und SciPy unvollständig bzw. pytorch-fid nicht installiert. Testabhängigkeiten wurden isoliert unter `/tmp/cut-eval-test-deps` ergänzt; die ursprüngliche Umgebung wurde nicht verändert. Für diesen isolierten Testlauf:

```bash
PYTHONPATH=/tmp/cut-eval-test-deps:. OPENBLAS_NUM_THREADS=1 python -m pytest tests/test_cut_evaluation.py -q
```

Ein vollständiges Training auf dem tatsächlichen medizinischen Datensatz und GPU-/Multi-GPU-Tests sind nicht Bestandteil der synthetischen Prüfung.

Prüfergebnis: 14 neue Tests und 5 bestehende FID-Tests bestanden (19 insgesamt), einschließlich echter Inception-FID. Der Gesamtlauf über alle vorhandenen Tests konnte `tests/test_prepare_doctr_cut_matched.py` nicht importieren: Die bereits referenzierte Datei `scripts/prepare_doctr_cut_matched.py` fehlt im Projekt. Dieses bestehende Problem wurde nicht durch eine Änderung am Datenexport überdeckt.

Der vollständige Smoke-Test `train.py` → `evaluate_cut.py` bestand ebenfalls auf einem synthetischen CPU-Datensatz: Loss-CSV, Validation-Vorschau, echte FID, best-Checkpoints, finale Testbilder und finales Vergleichsdiagramm wurden erzeugt. Testartefakte liegen unter `/tmp/cut-workflow-4q_tz8dj/`; die Trainingsdaten des Nutzers wurden nicht verändert.

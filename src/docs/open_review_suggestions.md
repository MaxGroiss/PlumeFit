Ausgewählte Findings aus AI-Code Reviews:
## Können fehler in Berechnung verursachen
### Schadstofffenster um den eigenen Peak werden nicht auf NULL_DATA geprüft
#### template_extraction.extract_plumes, poll_null
poll_null wird berechnet, aber nie verwendet. Geprüft wird nur das Fenster um den CO₂-Peak (in assess_pollutant_peak_centered). Das Fenster wird danach um den Schadstoffpeak neu geschnitten, also um bis zu −2…+4 s verschoben. Dieses verschobene Fenster prüft niemand mehr.
NaN oder ein Sensorausfall am Fensterrand kann in eine Vorlage gelangen.
-> poll_valid = co2_valid & (poll_statuses == VALID) & poll_in_bounds & ~poll_null und betroffene Zeilen auf NULL_DATA setzen.
### Schadstoff-qa_counts zählt Fahnen als VALID, deren CO₂-Fahne durchgefallen ist
#### template_extraction.extract_plumes, qa_counts=_qa_counts(poll_statuses)
poll_statuses enthält das Ergebnis der Schadstoff-QA für alle Zeilen, die die erste CO₂-Stufe bestanden haben. Fällt CO₂ in der zweiten Stufe durch (Mehrfachpeak, Abfall, Fläche), bleibt der Schadstoffstatus trotzdem VALID.
(synthetische Daten): NOₓ n_valid = 37, aber qa_counts['valid'] = 39. Im konstruierten Fall: 7 gültige, 20 als VALID gezählt.
Vor dem Zählen poll_statuses[~co2_valid & (poll_statuses == VALID)] auf einen eigenen Status setzen, z. B. CO2_INVALID, oder den CO₂-Status übernehmen
### qa_counts verliert Durchfahrten am Segmentrand, Summe ≠ n_isolated
#### cutout_isolated_plumes und extract_plumes
Liegt das LB-Fenster einer isolierten Durchfahrt außerhalb des Signals (Tagesanfang oder -ende), wird sie stillschweigend verworfen. WINDOW_EDGE wird nur für Peakfenster vergeben.
Statusarray über alle isolierten Durchfahrten führen, LB-Fenster außerhalb der Grenzen als WINDOW_EDGE. Die Schadstoffzählung umfasst nur lb_valid-Zeilen. Fahnen, die schon in der ersten CO₂-Stufe ausscheiden, fehlen dort ganz. Am besten beide Zählungen auf dieselbe Basis stellen, nämlich alle isolierten Durchfahrten.
### estimate_noise kann σ̂ = 0 liefern
#### noise_and_background.estimate_noise
Sind mehr als 50 % der fahrzeugfreien Werte identisch (grob quantisierte Schadstoffkanäle), ist MAD = 0. h_min = median, ρ_min = 0. In refine_positions ist die Schwelle κσ̂² = 0, also wird jede SSE-Verbesserung akzeptiert. Das ist eine mögliche Ursache für die langsame Positionsverfeinerung auf Schadstoffdaten.
ValueError oder Warnung bei σ̂ = 0. Alternativ eine Untergrenze aus der Messauflösung des Sensors.
### Extraktion schätzt Rauschen und Abfall-Schwelle auf Rohdaten inklusive Sensorausfällen
#### extract_plumes: resolve_qa_thresholds(config.co2_qa, co2_data - co2_bg_series, …), ebenso beim Schadstoff
fit_day setzt Fehlaufzeichnungen vorher auf NaN, die Extraktion nicht. Ein Ausfall (z. B. konstant 0) geht als großer negativer Residuumswert in die MAD und in die Kalibrierung von R_krit ein. Die Ausfallwerte selbst sind konstant und erzeugen daher nur am Rand Anstiege.
Leicht verschobene h_min, ρ_min und R_krit an Tagen mit langen Ausfällen. Extraktion und Fitting rechnen σ̂ auf unterschiedlicher Datenbasis.
residual = np.where(faulty_recording_mask(...), np.nan, co2_data) - co2_bg_series, analog zu fit_day.
### Gruppenamplitude steht bei jedem Mitglied – Summe über Durchfahrten zählt doppelt
#### fit_day → PassAmplitude.amplitude
Bei zusammengefassten Gruppen bekommt jedes Mitglied die Gesamtfläche der Gruppe. Wer in der Auswertung über alle Durchfahrten summiert (z. B. Tagesemission, Flottenmittel), zählt jede Gruppe n-fach.
Kein Codefehler, aber Auswertung immer über group deduplizieren. Alternativ ein Feld amplitude_share = amplitude / len(group) oder ein Flag is_group_representative. Docstring weist jetzt darauf hin.
## Konsistenz
### Randflag nur rechts, nicht links
#### fit_day: tail = (not seg.anchored_right) and gi == len(groups) - 1
Am Anfang des Messsegments (kein linker Ankerlauf) kann die erste Vorlage genauso abgeschnitten und per template_column hochgerechnet werden. Das wird nicht markiert.
Analog head = (not seg.anchored_left) and gi == 0 und beide in ein Flag edge_extrapolated zusammenführen. Dann auch Tab. 3.2 („Randflag“) anpassen.
### Fehlgeschlagene Segmente verschwinden aus pass_amplitudes
#### fit_day → SegmentRecord(..., "failed", ..., []), DayFitResult.pass_amplitudes
Durchfahrten in einem failed-Segment fehlen im Dictionary. In der Auswertung ist nicht unterscheidbar, ob eine Durchfahrt fehlgeschlagen ist oder nie übergeben wurde.
Für failed je Durchfahrt ein PassAmplitude mit amplitude = nan, detected = False anlegen.
### pos_lo ohne pos_hi (oder umgekehrt) führt zu einem TypeError tief in refine_positions
Vorschlag: In fit_day am Anfang: if (pos_lo is None) != (pos_hi is None): raise ValueError(...).
### n_anchor_min ist an die Abtastrate gekoppelt und bei den Standardwerten wirkungslos
Problem: Ein Segment enthält immer seine begrenzenden Ankerläufe vollständig. Jeder Ankerlauf ist ≥ min_anchor_run = 5 s lang. Bei Δt ≤ 0,5 s sind das ≥ 10 Samples, also hat jedes Segment mit mindestens einem Ankerlauf schon n_anchor ≥ n_anchor_min. Dann greift anchored_fixed nur für Segmente ganz ohne Ankerlauf, und 2 · n_anchor_min für die lineare Baseline ist bei beidseitiger Verankerung automatisch erfüllt. Bei Δt = 1 s verhält sich dieselbe Config anders.
Vorschlag: Als Dauer angeben (min_anchor_time: timedelta64) oder die Bedingung direkt über die Anzahl der Ankerläufe formulieren.
### except ValueError in fit_day fängt alles
Problem: Gedacht für „Vorlage passt nicht ins Segment“ und „zu wenige Samples“. Jeder andere ValueError (z. B. ein Indexfehler in der Gruppierung) wird ebenfalls still als failed verbucht.
Vorschlag: Eigene Exception (class SegmentNotIdentifiable(ValueError)) in fit_mode_l werfen und nur diese fangen, oder den Fehlertext im SegmentRecord mitspeichern.
### min_gap gegen Peakfenster und Ausschlussfenster
Problem: Isoliert heißt ≥ 30 s Abstand. Für die Rauschschätzung gilt ein Fahrzeug aber bis 35 s nach dem Trigger als wirksam (peak_search_after + window_after_peak). Der Ausläufer des Vorgängers kann im Nullungsbereich landen, deshalb war area_plausibility_check nötig.
Vorschlag: In __post_init__ min_gap ≥ window_before_peak + window_after_peak prüfen, oder den Standardwert von min_gap auf ≥ 35 s setzen. Die Bedingung steht schon im Docstring.
### Mehrfachpeak-Kriterium CO₂ ≠ Schadstoff
Problem: Beim Schadstoff muss ein zweiter Peak zusätzlich ≥ h_min hoch sein (height_valid), bei CO₂ reicht die Prominenz.
Vorschlag: Entweder angleichen oder den Grund in Kommentar und Text festhalten. Möglicher Grund: Ein kleiner zweiter CO₂-Peak zeigt trotzdem eine Überlappung an.
 


SPLIT_KEY   = "exhaust_side_assumed" # -> Template Schlüssel
MIN_N       = 30 # -> Vorlagen werden pre Checked ob mindestens MIN_N Passes zu Vorlage beitragen

# Fallback wenn ein Pass den Key Nicht hat -> Allgemeines Template also über alle Kategorien

# Idee ist, dass man gruppierte CSV erstellt -> So wie man die Templates aufteilen will -> Exhaust Side + etc Key.
# Das Fitting liest diese ein und mappt dann über Vehicle Mapping automatisch auf die LB Passes je nach Key Value
# das aus der CSV erstellte Template

# 1 Einlesen / Prüfen -> CombinedResult Pro Kanal erstellen

# 2 Zuweisung über Mapping -> Dieser Pass bekommt dieses Template

# 3 Fitting
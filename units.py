UNIT = "ms"                 # keep "ms" since your labels are in ms
DELTA = 20.0               # Huber delta ≈ 20 ms
LOW_RR, HIGH_RR = 300.0, 1800.0   # physiologic RR bounds in ms

# sensible SSM priors in ms
RR0_INIT = 900.0           # ~67 bpm
P0_RR_VAR = 100.0**2       # large-ish prior variance (ms^2)
P0_DPAT_VAR = 50.0**2
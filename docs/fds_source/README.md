# FDS source: the wildfire (level-set) code

`vege.f90` is copied unchanged from the official NIST FDS 6.11.1 release:
https://github.com/firemodels/fds/blob/FDS-6.11.1/Source/vege.f90
(FDS is US government software and in the public domain.)

It holds everything the simulations here use to spread fire:

| Lines | What it does |
|---|---|
| 275–279 | Wind-factor constants from the fuel fineness |
| 424 | Wind reduced from 6.1 m to flame height |
| 441 | Wind boost (Rothermel wind factor) |
| 499 | Head-fire speed = no-wind speed × (1 + wind boost) |
| 761–873 | Egg-shaped spread: speed in every direction |
| 881–930 | Moving the fire edge one time step (level set) |
| 993–1225 | Rothermel no-wind spread speed, including the 13 fuel types (1010–1063) |

Default input values (moisture 3%, wind height 6.1 m, etc.) are in `Source/read.f90`.

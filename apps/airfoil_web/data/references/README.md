# Airfoil validation data

These small numeric datasets ship with Airfoil Lab so geometry and experimental
Cp curves are available offline. They are reference measurements, not solver
results. No journal paper is redistributed.

* `rae2822_agard.dat`: AGARD AR-138 Table 6.1 ordinates from the
  [NASA NPARC geometry archive](https://www.grc.nasa.gov/www/wind/valid/raetaf/geom.txt),
  converted to a trailing-edge/leading-edge/trailing-edge contour.
* `rae2822_case9_exp_full.csv`: 52 upper and 50 lower pressure taps transcribed
  from Cook, McDonald and Firmin, AGARD AR-138 (1979), Appendix A6, Table 6.7.
  [Case 9 CFD conditions](https://cfd.ku.edu/hiocfd/case_c2.2.html) are corrected
  free-air values (M=0.734, alpha=2.79°, Re=6.5 million, trips=0.03c), distinct
  from the original tunnel settings (M=0.730, alpha=3.19°).
* `rae2822_case6_cp.gen`: unchanged [NASA experimental file](https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf04/cp.exp.gen).
  Its second column is **minus Cp**; the reader reverses that sign.
* `onera_oat15a.dat`: normalized ONERA OAT15A coordinates from the
  [DPW geometry release, 5 September 2024](https://media.githubusercontent.com/media/dpw-committee/aiaa-dpw/main/Geometry/ONERA-OAT15A-090524.zip).
* `onera_oat15a_cp.dat`: unchanged Cp.dat from the
  [ONERA experimental dataset](https://dpw.larc.nasa.gov/DPW8/ONERA_OAT15A/Experimental_Data/onera-oat15a.zip).
  Mean Cp at six angles: 2.5°, 3°, 3.1°, 3.25°, 3.5°, 3.9°.
  Mach=0.73, Re=3 million; experimental trips=0.07c.
  The [DPW conditions](https://www.aiaa-dpw.org/TestCases/testcases.html) prescribe
  static temperature 271 K and unsteady analysis for the extended polar.
  No measured force coefficients are supplied by this dataset.

`naca0012_ladson.dat` contains the unchanged Ladson force-data file from the
[NASA Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/naca0012_val.html)
(NASA TM 4074, 1988). The application plots its 17-point, 80-grit tripped series
at Mach 0.15 and Re=6 million. The pressure solver is an incompressible,
fully turbulent approximation, and the built-in NACA geometry differs from
the TMR scaled sharp-edge definition. Near-stall comparisons are exploratory:
the experiments have significant three-dimensional effects. No experimental
moment or pressure-distribution values are invented for this force dataset.

## ONERA acknowledgment and redistribution conditions

ONERA is acknowledged as the provider of the ONERA OAT15A geometry and data.
Under the [ONERA/AIAA conditions](https://www.aiaa-dpw.org/experiment.html#oat),
ownership of the geometry stays with ONERA. Use the name **ONERA OAT15A**.
Every onward sharing must acknowledge ONERA and carry these same conditions.
These data are not covered by this repository's software license.
The associated AIAA paper is for personal use and is not included.

# Workbench interface reference

The compact navigation, dark interface colours and six contour palettes
in `static/workbench.css` and `static/palettes.js` are adapted from
[JFEM-Boxer](https://github.com/r-acad/JF_2026_05_25/tree/main/APPS/JFEM-Boxer),
part of OpenJFEM, copyright its contributors, licensed under AGPL-3.0.
The source revision was [`010f712`](https://github.com/r-acad/JF_2026_05_25/tree/010f712e38fe24866d2b09d3080383580e8e0e26).
The files consulted were `web/style.css`, `web/project_menu.css`,
`web/model_workspace.css` and `web/color_scales.js` on 1 November 2026.
The signed palette, editable field limits and Aero Lab aerodynamic frames also
follow `web/result_display.js`, `web/viewport-legends.js`,
`web/aerodynamic_reference.js`, `web/aerodynamic_view_svg.js`,
`web/aero_display.js` and `src/aerodynamic_axes.jl` at that revision.
OpenAeroLab is distributed under the same licence; see the repository `LICENSE`.

`static/boxer_legends.js` is adapted from `web/viewport-legends.js` at that
revision. Airfoil Lab uses its palette selector, editable limits and sliders.

Navigation, study/result handling and installers remain implemented for this
repository's own application and solver environments. The setup/launch workflow
follows OpenJFEM's root setup and Windows/Linux/Mac launcher organization.

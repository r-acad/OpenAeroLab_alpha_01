# Makie exports

`PostprocessLauncher.jl` runs the renderer in this separate Julia environment,
so web solver workers do not import Makie into their simulation modules.
The root setup installs and verifies this environment automatically; see the
[installation instructions](../../README.md#install-and-launch).

The request accepts `source_directory`, `destination_directory`, `title`,
`report`, `charts`, `render_3d`, `chart_format` (`png`, `svg`, `pdf`),
`surface_quantity` (`cp`, `pressure`, `shear`), `camera` (`perspective`, `top`,
`front`, `side`), and `image_width`/`image_height`. Reports are HTML. The 3D
render is a PNG by default; `render_format` can also be `svg` or `pdf`.
The destination manifest lists generated artifacts and unavailable-data notes.

Saved iteration CSVs, LBMP surface pressure files, binary STL, AMR geometry,
and numeric LBM cuts are discovered under the source directory. Optional
`airfoil_results` points to normalized JSON with `charts` containing `name`,
`title`, `xlabel`, `ylabel`, optional `yreverse`, and `series` of
`{label,x,y}`. Geometry-only renders are labeled explicitly. A report never
claims that a saved iteration history establishes physical convergence.

/* Airfoil Lab's workflow help and contextual help. No solver settings live here. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const descriptions={
    "backend":["Solver","Choose the solver first: every step then shows only the inputs that solver uses. Low-speed 2D CFD is the default for clean airfoils and Gurney flaps. Compressible CFD covers Mach 0.02 to 0.98; Species / cavitation CFD covers water and vapor; LBM 3D extrudes the section along a periodic span; JXfoil and XFOIL are panel methods. Comparison runs start two or three solvers on the same case."],
    "source-kind":["Airfoil source","Open an XFOIL coordinate file for your own section, or choose a built-in airfoil."],
    "dat-file":["Airfoil coordinate file","Choose an XFOIL .dat text file. An optional title can precede x y coordinate pairs. Both a continuous contour and the two-surface count format are accepted. The chord is normalized to one; a saved study embeds the coordinates."],
    "reynolds":["Reynolds number","Re = freestream speed × chord / kinematic viscosity. Enter the value for your experiment or operating point. Low-speed CFD uses this to set viscosity; the default is 3 million."],
    "alpha":["Angle of attack","The angle in degrees between the incoming flow and the airfoil chord. Positive angles raise the leading edge relative to the flow. Near stall, inspect the flow and convergence before interpreting forces."],
    "mach":["Mach number","Freestream speed divided by the speed of sound. Low-speed CFD is incompressible and ignores Mach; this input appears only for solvers that use it."],
    "ncrit":["Transition Ncrit","The disturbance amplification threshold used by XFOIL and JXfoil transition models. It does not control any CFD turbulence model."],
    "gurney-enabled":["Gurney flap","Add a straight, thin plate perpendicular to the chord at the trailing edge. The lower-side default extends below the airfoil. Low-speed flaps require finite trailing-edge coordinates: a closed trailing edge is refused, preserving your source section. The clean and flap cases use matching solver settings. Compare at the same Reynolds number and angle."],
    "gurney-height":["Gurney flap height","Height divided by chord: 0.01 is a 1% chord flap. For a symmetric plate, the entered height is the total, split equally above and below the trailing edge."],
    "gurney-mode":["Gurney flap side","Lower places the flap below the trailing edge, upper above it, and symmetric places half the entered height on each side. The plate is normal to the chord."],
    "gurney-refine":["Flap mesh refinement","Off uses the section's own C+H mesh at the flap tip; it is the default. Standard and Fine add smaller cells near the tip and the wake, to test whether forces depend on mesh resolution there."],
    "preview-resolution":["Mesh resolution","The one resolution of the compressible and Species meshes: Create mesh and Run both use it. Standard uses curvature-aware leading-edge spacing. The actual block dimensions and total cells appear after Create mesh."],
    "trailing-edge-thickness":["CFD trailing-edge thickness","Optional override of the trailing-edge thickness used by the CFD mesh, as a fraction of chord. Leave blank to keep the section as given. XFOIL and JXfoil keep the source section."],
    "rans-method":["Steady solution method","How steady compressible CFD approaches a solution. Standard explicit is the ordinary choice. Implicit Newton (CPU) is a development result, not validated: a Newton–Krylov march with SST on the CPU in double precision, always on the marched regular mesh, starting cold on three mesh levels at every angle. It is meant for steady compressible cases with a sharp or a blunt trailing edge and takes minutes per case. CFD steps limits its fine level (at most 600, default 300). Its point is marked completed only when lift and drag settle on the fine mesh; otherwise it is marked not converged and the fields remain a diagnostic state. The GPU switch, smart stop, the scaled mass ceiling, the SFD filter precision, the RK weights and streamline alignment do not apply to it. Implicit residual smoothing is an experimental variant of the explicit iteration."],
    "lowspeed-resolution":["Low-speed mesh","Standard is the recommended mesh. Fine subdivides the C+H mesh of a blunt trailing edge once more (four times the cells); on a sharp trailing edge it is the same mesh as Standard and is not offered. The mesh is built when you Run, or use Create mesh to inspect its quality first."],
    "lowspeed-start":["Starting flow","Automatic is the gradual start, used for clean and flap cases on the C+H mesh. Projected start is for checking sensitivity to the starting state, particularly near stall. It applies to C+H only and is not sent for a sharp trailing edge."],
    "lowspeed-mut-base":["Base turbulence treatment","Automatic is the committed treatment, used for clean and flap cases. Extrapolated is for controlled C+H comparisons. It applies to C+H only and is not sent for a sharp trailing edge."],
    "lowspeed-tu":["Freestream turbulence intensity","The turbulent velocity fluctuation level as a percentage of freestream speed. The default is 1%. Change it when a measured inlet condition is available."],
    "lowspeed-mut-ratio":["Freestream eddy viscosity ratio","The ratio of modeled turbulent viscosity to molecular viscosity in the freestream. The default is 10. Match it between clean and flap comparisons."],
    "lowspeed-steady-tol":["Momentum steady limit","The absolute steady momentum residual limit. The default is 1e-6. Larger limits may stop earlier; compare forces and fields against a tighter result before adopting them."],
    "lowspeed-turb-steady-tol":["Turbulence steady limit","The steady residual limit for both turbulence equations. The default is 1e-7. A nearly constant lift or drag history alone does not show that these equations have settled."],
    "lowspeed-gpu":["GPU execution","Use a compatible NVIDIA GPU instead of the CPU. The job waits if another solver owns the GPU. CPU execution is the default; both paths retain the same convergence checks."],
    "sweep-enabled":["Multiple angles","Run a sequence of angles using the same geometry and flow settings. Completed points are retained if a later angle is stopped. Near stall, sweep direction and starting state can affect a steady branch."],
    "create-mesh":["Create mesh","Preview the same mesh that Run will use without starting the flow calculation. Inspect the trailing edge, flap tip, and mesh-quality report. Running a case also creates the mesh automatically."],
    "cancel-button":["Cancel run","Requests a stop for the current job. Completed points, recorded fields, and logs remain available. The current operation may need to finish before the job reaches Cancelled."],
    "field-kind":["Flow field quantity","Choose a saved flow quantity. Low-speed velocity is shown relative to freestream speed; the app uses the units recorded with the field. A saved field can be inspected even if the run did not converge."],
    "show-reference":["Validation reference","Show measured data when matching reference conditions are available. A reference overlay does not establish that a computed case has converged or is independent of mesh resolution."]
  };
  // The workflow in the order a user works; the list numbers are the step numbers of the tab row. The Help step shows
  // the short form of the same steps.
  const first="First choose the solver, above the steps. Every step then shows only what that solver uses. Low-speed 2D CFD is the default; Compressible CFD, Species / cavitation CFD, LBM 3D, JXfoil, XFOIL and comparison runs are the other choices.";
  const workflow=[
    "Airfoil: open an XFOIL .dat coordinate file or pick a built-in section. A Gurney flap is added here, for the low-speed CFD and LBM 3D.",
    "Flow: enter Reynolds number and angle, and Mach where the solver uses it. Switch on the alpha sweep for a polar.",
    "Mesh: choose the resolution. Create mesh builds and shows the mesh that Run will use; Run builds it anyway. A panel method has no mesh.",
    "Solver settings: the method, the GPU switch and the stopping limits. The defaults suit an ordinary run.",
    "Run and outputs: choose the saved files and press Run analysis. Run and Cancel also stay below every step; Cancel keeps the completed results.",
    "Results: forces, surface pressure, fields, the polar and histories appear as data arrive. Check Diagnostics before using a force or a field."
  ];
  const notes=[
    "Advanced groups hold rarely needed settings. They stay closed unless a loaded study, saved job or preset uses one of them.",
    "Study saves and reopens a setup; a saved study makes its own folder the default for new date-and-time run folders. Validation loads published comparison cases. Display holds the palette, grid and probe.",
    "Implicit Newton (CPU): select Compressible CFD, then in Solver settings choose the steady solution method Implicit Newton (CPU). It is for steady compressible cases on the CPU, with a sharp or a blunt trailing edge, in minutes per case, and always runs on the Marched regular mesh (one block for a closed edge, a C block plus a strip for a blunt one). Development result, not validated."
  ];
  let dialog;
  function openDialog(title,build){
    if(!dialog){
      dialog=document.createElement("dialog");dialog.className="field-help-dialog";dialog.id="field-help-dialog";
      dialog.setAttribute("aria-labelledby","field-help-title");
      dialog.innerHTML='<h2 id="field-help-title"></h2><div id="field-help-body"></div><button type="button" class="secondary-button">Close</button>';
      dialog.querySelector("button").addEventListener("click",()=>dialog.close());document.body.append(dialog);
    }
    $("field-help-title").textContent=title;
    const body=$("field-help-body");body.replaceChildren();build(body);
    dialog.showModal();dialog.querySelector("button").focus();
  }
  const paragraph=text=>{const node=document.createElement("p");node.textContent=text;return node;};
  function showHelp(title,text){openDialog(title,body=>body.append(paragraph(text)));}
  function showWorkflow(){
    openDialog("Airfoil Lab workflow",body=>{
      const list=document.createElement("ol");list.className="workflow-steps";
      for(const step of workflow){const item=document.createElement("li");item.textContent=step;list.append(item);}
      body.append(paragraph(first),list,...notes.map(paragraph));
    });
  }
  function attach(id,title,text){
    const control=$(id);if(!control)return;
    control.title=text;
    const label=control.closest("label"),heading=label?.querySelector(":scope > span");
    if(!heading||heading.querySelector(".info-button"))return;
    const button=document.createElement("button");button.type="button";button.className="info-button";button.textContent="i";
    button.title=text;button.setAttribute("aria-label",`Information: ${title}`);button.setAttribute("aria-haspopup","dialog");
    button.addEventListener("click",event=>{event.preventDefault();event.stopPropagation();showHelp(title,text);});
    heading.append(button);
  }
  for(const [id,[title,text]] of Object.entries(descriptions))attach(id,title,text);
  window.AirfoilContextHelp={refresh(){
    for(const [id,[,text]] of Object.entries(descriptions)){const control=$(id);if(control&&!control.disabled)control.title=text;}
  }};
  document.querySelectorAll("[data-airfoil-workflow-help]").forEach(button=>button.addEventListener("click",showWorkflow));
})();

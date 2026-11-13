/* Airfoil Lab's primary workflow and contextual help. No solver settings live here. */
(() => {
  "use strict";
  const $=id=>document.getElementById(id);
  const descriptions={
    "source-kind":["Airfoil source","Open an XFOIL coordinate file for your own section, or choose a built-in airfoil. The default analysis is low-speed 2D CFD."],
    "clear-saved-flow-overrides":["Clear saved overrides","Remove only additional settings from a saved request and return them to automatic defaults for the next run. Keep the current airfoil, visible flow conditions and flap controls. A running job continues unchanged; the study is marked as changed."],
    "dat-file":["Airfoil coordinate file","Choose an XFOIL .dat text file. An optional title can precede x y coordinate pairs. Both a continuous contour and the two-surface count format are accepted. The chord is normalized to one; a saved study embeds the coordinates."],
    "reynolds":["Reynolds number","Re = freestream speed × chord / kinematic viscosity. Enter the value for your experiment or operating point. Low-speed CFD uses this to set viscosity; the default is 3 million."],
    "alpha":["Angle of attack","The angle in degrees between the incoming flow and the airfoil chord. Positive angles raise the leading edge relative to the flow. Near stall, inspect the flow and convergence before interpreting forces."],
    "mach":["Mach number","Freestream speed divided by the speed of sound. Low-speed CFD is incompressible and ignores Mach; this input appears only for solvers that use it."],
    "ncrit":["Transition Ncrit","The disturbance amplification threshold used by XFOIL and JXfoil transition models. It does not control the fully turbulent low-speed CFD model."],
    "gurney-enabled":["Gurney flap","Add a straight, thin plate perpendicular to the chord at the trailing edge. The lower-side default extends below the airfoil. Low-speed flaps require finite trailing-edge coordinates: a closed trailing edge is refused, preserving your source section. The clean and flap cases use matching solver settings. Compare at the same Reynolds number and angle."],
    "gurney-height":["Gurney flap height","Height divided by chord: 0.01 is a 1% chord flap. For a symmetric plate, the entered height is the total, split equally above and below the trailing edge."],
    "gurney-mode":["Gurney flap side","Lower places the flap below the trailing edge, upper above it, and symmetric places half the entered height on each side. The plate is normal to the chord."],
    "gurney-refine":["Flap mesh refinement","Adds smaller cells near the flap tip and wake. Keep the recommended setting for an ordinary run. Use another level to test whether forces depend on mesh resolution."],
    "backend":["Solver","Low-speed 2D CFD is the default for clean airfoils and Gurney flaps. Other methods remain available for compressible flow, panel comparisons, and research. Each method has its own range of supported physics."],
    "cfd-mesh":["Compressible mesh profile","Keep Refined shared mesh for an ordinary case. Uniform A+ quality and fixed-spacing profiles are research selections with their own A+ topology. An explicit saved profile is preserved when you reopen, save or run the study; these profiles are not validated defaults."],
    "implicit-globalization":["Update acceptance method","Optional Newton–Krylov research setting. Automatic keeps the steady update. Backward Euler adds a pseudo-time term and allows a higher pseudo CFL ceiling. Final equation checks remain unchanged; faster airfoil convergence is not established."],
    "implicit-linear-maxiter":["Maximum linear iterations","Optional total linear iterations per Newton update. Blank uses the basis size. A larger total can restart the basis without storing a larger basis; the solver still checks its workspace budget."],
    "implicit-max-cfl":["Maximum pseudo CFL","Optional upper limit for the attempted steady update size. It must be at least the initial pseudo CFL. Blank retains the solver default; a higher ceiling does not guarantee a larger accepted update."],
    "implicit-physical-fraction":["Physical update fraction","Optional Newton–Krylov research limit on each physical state update, from 0.01 to 0.5. Blank retains 0.2. Smaller fractions can avoid unphysical trial states but may require more iterations."],
    "implicit-linear-tolerance":["Linear relative tolerance","Optional relative error limit for the inner linear solve. Blank retains 0.01. This does not change the final flow or turbulence convergence checks."],
    "lowspeed-resolution":["Low-speed mesh","Automatic selects the recommended grid for the section. Standard and Fine are explicit settings for a mesh comparison. The mesh is built when you Run, or use Create mesh to inspect its quality first."],
    "lowspeed-topology":["Low-speed topology","Automatic retains the airfoil trailing edge and chooses C+H for a finite gap or sharp C for a closed edge. Gurney flaps require a finite gap. Clean and flap cases share the same settings. Override the topology only for a controlled numerical comparison."],
    "lowspeed-start":["Starting flow","Automatic uses the same gradual start for clean and flap cases on the native C+H mesh. The explicit alternatives are for checking sensitivity to starting state, particularly near stall; they apply to C+H only."],
    "lowspeed-mut-base":["Base turbulence treatment","Automatic uses the same base treatment for clean and flap cases. The explicit alternatives are for controlled C+H comparisons and preserve settings from earlier saved studies."],
    "lowspeed-tu":["Freestream turbulence intensity","The turbulent velocity fluctuation level as a percentage of freestream speed. The automatic default is 1%. Change it when a measured inlet condition is available."],
    "lowspeed-mut-ratio":["Freestream eddy viscosity ratio","The ratio of modeled turbulent viscosity to molecular viscosity in the freestream. The automatic default is 10. Match it between clean and flap comparisons."],
    "lowspeed-steady-tol":["Momentum steady limit","The absolute steady momentum residual limit. The default is 1e-6. Larger limits may stop earlier; compare forces and fields against a tighter result before adopting them."],
    "lowspeed-turb-steady-tol":["Turbulence steady limit","The steady residual limit for both turbulence equations. The default is 1e-7. A nearly constant lift or drag history alone does not show that these equations have settled."],
    "lowspeed-gpu":["GPU execution","Use a compatible NVIDIA GPU instead of the CPU. The job waits if another solver owns the GPU. CPU execution is the default; both paths retain the same convergence checks."],
    "sweep-enabled":["Multiple angles","Run a sequence of angles using the same geometry and flow settings. Completed points are retained if a later angle is stopped. Near stall, sweep direction and starting state can affect a steady branch."],
    "create-mesh":["Create mesh","Preview the same mesh that Run will use without starting the flow calculation. Inspect the trailing edge, flap tip, and mesh-quality report. Running a case also creates the mesh automatically."],
    "cancel-button":["Cancel run","Requests a stop for the current job. Completed points, recorded fields, and logs remain available. The current operation may need to finish before the job reaches Cancelled."],
    "field-kind":["Flow field quantity","Choose a saved flow quantity. Low-speed velocity is shown relative to freestream speed; the app uses the units recorded with the field. A saved field can be inspected even if the run did not converge."],
    "show-reference":["Validation reference","Show measured data when matching reference conditions are available. A reference overlay does not establish that a computed case has converged or is independent of mesh resolution."]
  };
  let dialog;
  function showHelp(title,text){
    if(!dialog){
      dialog=document.createElement("dialog");dialog.className="field-help-dialog";dialog.id="field-help-dialog";
      dialog.setAttribute("aria-labelledby","field-help-title");
      dialog.innerHTML='<h2 id="field-help-title"></h2><p id="field-help-text"></p><button type="button" class="secondary-button">Close</button>';
      dialog.querySelector("button").addEventListener("click",()=>dialog.close());document.body.append(dialog);
    }
    $("field-help-title").textContent=title;$("field-help-text").textContent=text;
    dialog.showModal();dialog.querySelector("button").focus();
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
  document.querySelectorAll("[data-airfoil-workflow-help]").forEach(button=>button.addEventListener("click",()=>{
    window.LudwigWorkbench?.closeMenus();
    showHelp("Low-speed airfoil workflow","1. Airfoil: open an XFOIL .dat coordinate file. 2. Flow: enter Reynolds number and angle. 3. Flap: add a Gurney plate only if required. 4. Run analysis. The app selects numerical defaults and creates the grid; Mesh opens its optional settings and inspection. Solver contains stopping limits, Files contains output choices, and Study saves or reopens the setup. A saved study makes its own folder the default for new date-and-time run folders. Results and its available views appear when data arrive. Check Diagnostics before using forces or fields. Cancel run retains completed results.");
  }));
})();

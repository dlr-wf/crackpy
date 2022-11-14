"""

    Example script:
        Simulation of compact tension (CT) specimen with pyansys

    Needed:
        - ANSYS installation

    Output:
        - folder CT_Simulation_Output containing nodemap file ('Nodemap.txt') and plots

"""

# Imports
import os
import time

from ansys.mapdl.core import launch_mapdl
import ansys.mapdl.core.errors

from crackpy.ansys_simulation.models import CTSimulation
from crackpy.ansys_simulation.utils import delete_ansys_leftovers
from crackpy.structure_elements.material import Material


OUTPUT_PATH = 'CT_Simulation_Output'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Parameters
FORCE = 10000
ALPHA = 0.5
WIDTH = 75
THICKNESS = 10
ESIZE = 0.5
REF_WINDOW = 8

# Material
material = Material(
    name='AA2024-T3',
    E=72000,
    nu_xy=0.33,
    sig_yield=350,
    plane_strain=False
)

while True:

    try:
        # Start MAPDL
        mapdl = launch_mapdl(additional_switches='-smp', run_location=OUTPUT_PATH, nproc=5)

        # Initialize CT simulation
        ansys_simulation = CTSimulation(mapdl)

        ansys_simulation.set_parameters(
            material=material,
            width=WIDTH,
            thickness=THICKNESS,
            force=FORCE,
            alpha=ALPHA,
            esize=ESIZE,
            ref_window=REF_WINDOW
        )

        # Key points and Mesh
        ansys_simulation.set_key_points()
        ansys_simulation.set_mesh()

        # Boundary conditions
        ansys_simulation.set_boundary_conditions()

        # Solve
        ansys_simulation.solve()

        # Plot
        ansys_simulation.plot(yield_stress=material.sig_yield)

        # Export
        ansys_simulation.export_nodemap()

    # Catch exceptions
    except ansys.mapdl.core.errors.LockFileException:
        delete_ansys_leftovers(ansys_folder=OUTPUT_PATH)
        continue

    except ansys.mapdl.core.errors.MapdlExitedError:
        print('Mapdl Session Terminated. Retrying...')
        delete_ansys_leftovers(ansys_folder=OUTPUT_PATH)
        continue

    except OSError:
        print('OSError. Retrying...')
        continue

    finally:
        mapdl.exit()
        time.sleep(5)
        delete_ansys_leftovers(ansys_folder=OUTPUT_PATH)

    break

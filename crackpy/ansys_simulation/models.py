import os

from crackpy.structure_elements.material import Material


class AnsysSimulation:
    """Class for Simulations in ANSYS MECHANICAL.

    Methods:
        * export_nodemap - saves Ansys node data into txt-file 'Nodemap'
        * plot - plots mesh, displacements, and von-Mises strain
        * solve - solution of PDE

    """

    def __init__(self, mapdl):
        """Class for Simulations in ANSYS MECHANICAL.

        Args:
            mapdl: PyAnsys MAPDL instance

        """
        self.mapdl = mapdl

        # Initialization
        self.mapdl.run("/BATCH")
        self.mapdl.finish()
        self.mapdl.clear()
        self.mapdl.prep7()
        self.mapdl.seltol(0.01)

    def export_nodemap(self) -> None:
        """Saves the current Ansys node data into a txt-file called 'Nodemap'."""

        # Clear arrays
        self.mapdl.run("*DEL,coords_1")
        self.mapdl.run("*DEL,coords_2")
        self.mapdl.run("*DEL,header")

        # Change to post-processing
        self.mapdl.post1()
        self.mapdl.set("LAST")
        self.mapdl.allsel()

        # Get total number of nodes
        self.mapdl.get("NodeCount_1", "NODE", "", "COUNT")

        # Get coordinates, displacements, strains, and stresses
        self.mapdl.run("*VGET,dEPTOXs,NODE,,EPTO,X")
        self.mapdl.run("*VGET,dEPTOYs,NODE,,EPTO,Y")
        self.mapdl.run("*VGET,dEPTOXYs,NODE,,EPTO,XY")
        self.mapdl.run("*VGET,dEPTOEQVs,NODE,,EPTO,EQV")
        self.mapdl.run("*VGET,dUXs,NODE,,U,X")
        self.mapdl.run("*VGET,dUYs,NODE,,U,Y")
        self.mapdl.run("*VGET,dUZs,NODE,,U,Z")
        self.mapdl.run("*DIM,coords_1,ARRAY,NodeCount_1,14")  # Prepare Output-File
        self.mapdl.run("*VGET,coords_1(1,1),NODE,,NLIST")  # Node index
        self.mapdl.run("*VGET,coords_1(1,2),NODE,coords_1(1,1),LOC,X")  # Undeformed coordinates X
        self.mapdl.run("*VGET,coords_1(1,3),NODE,coords_1(1,1),LOC,Y")  # Undeformed coordinates Y
        self.mapdl.run("*VGET,coords_1(1,4),NODE,coords_1(1,1),LOC,Z")  # Undeformed coordinates Z
        self.mapdl.run("*VGET,coords_1(1,5),NODE,coords_1(1,1),U,X")  # Displacements X
        self.mapdl.run("*VGET,coords_1(1,6),NODE,coords_1(1,1),U,Y")  # Displacements Y
        self.mapdl.run("*VGET,coords_1(1,7),NODE,coords_1(1,1),U,Z")  # Displacements Z
        self.mapdl.run("*VOPER,coords_1(1,8),dEPTOXs,MULT,100.0")  # Total strains X
        self.mapdl.run("*VOPER,coords_1(1,9),dEPTOYs,MULT,100.0")  # Total strains Y
        self.mapdl.run("*VOPER,coords_1(1,10),dEPTOXYs,MULT,0.5")  # Total strains XY
        self.mapdl.run("*VOPER,coords_1(1,11),dEPTOEQVs,MULT,100.0")  # Total strains XY
        self.mapdl.run("*VGET,coords_1(1,12),NODE,coords_1(1,1),S,X")  # Stress X
        self.mapdl.run("*VGET,coords_1(1,13),NODE,coords_1(1,1),S,Y")  # Stress Y
        self.mapdl.run("*VGET,coords_1(1,14),NODE,coords_1(1,1),S,XY")  # Stress XY

        # Define header
        self.mapdl.run("*DIM,header,string,10,14")
        self.mapdl.run("header(1,1)='index'")
        self.mapdl.run("header(1,2)='x_undf'")
        self.mapdl.run("header(1,3)='y_undf'")
        self.mapdl.run("header(1,4)='z_undf'")
        self.mapdl.run("header(1,5)='ux'")
        self.mapdl.run("header(1,6)='uy'")
        self.mapdl.run("header(1,7)='uz'")
        self.mapdl.run("header(1,8)='eps_x'")
        self.mapdl.run("header(1,9)='eps_y'")
        self.mapdl.run("header(1,10)='eps_xy'")
        self.mapdl.run("header(1,11)='eps_eqv'")
        self.mapdl.run("header(1,12)='s_x'")
        self.mapdl.run("header(1,13)='s_y'")
        self.mapdl.run("header(1,14)='s_xy'")

        # Write into file
        with self.mapdl.non_interactive:
            self.mapdl.run("*CFOPEN,Nodemap,txt")
            self.mapdl.run("*VWRITE,header(1,1),header(1,2),header(1,3),header(1,4),header(1,5),header(1,6),"
                           "header(1,7),header(1,8),header(1,9),header(1,10),"
                           "header(1,11),header(1,12),header(1,13),header(1,14)")
            self.mapdl.run("('#',A11,13(';',A12))")
            self.mapdl.run("*VWRITE,coords_1(1,1),coords_1(1,2),coords_1(1,3),coords_1(1,4),coords_1(1,5),"
                           "coords_1(1,6),coords_1(1,7),coords_1(1,8),coords_1(1,9),coords_1(1,10),"
                           "coords_1(1,11),coords_1(1,12),coords_1(1,13),coords_1(1,14)")
            self.mapdl.run("(13(F12.6,';')F12.6)")
            self.mapdl.run("*CFCLOSE")
            self.mapdl.run("/GOPR")

    def plot(self, yield_stress: float) -> None:
        """Plots the mesh, displacements, and von-Mises strain.

        Args:
            yield_stress: of the material (used as upper limit for stress plot)

        """
        self.mapdl.post1()
        self.mapdl.esel('S', 'TYPE', vmin=1)  # mask out holes

        # plot element mesh
        self.mapdl.eplot(cpos='xy', savefig=os.path.join(self.mapdl.directory, 'mesh.png'))

        # colorbar settings
        sbar_kwargs = dict(
            title_font_size=20,
            label_font_size=16,
            n_labels=5,
            font_family="arial",
            color="black"
        )

        # plot nodal von-Mises stress
        plotter = self.mapdl.post_processing.plot_nodal_eqv_stress(
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="jet",
            return_plotter=True,
        )
        plotter.update_scalar_bar_range([0, yield_stress])
        plotter.save_graphic(os.path.join(self.mapdl.directory, 'stress_eps_vm.svg'))

        # plot nodal y-displacement
        self.mapdl.post_processing.plot_nodal_displacement(
            component='Y',
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="jet",
            savefig=os.path.join(self.mapdl.directory, 'y_displacement.png')
        )
        # plot nodal x-displacement
        self.mapdl.post_processing.plot_nodal_displacement(
            component='X',
            cpos='xy',
            background='white',
            scalar_bar_args=sbar_kwargs,
            show_axes=True,
            n_colors=256,
            off_screen=True,
            cmap="jet",
            savefig=os.path.join(self.mapdl.directory, 'x_displacement.png')
        )

    def solve(self):
        self.mapdl.run("/SOLU")
        self.mapdl.solve()


class MTSimulation(AnsysSimulation):
    """Class for MT Specimen Simulation in ANSYS MECHANICAL.

    Methods:
        * set_parameters - variables and parameters for Ansys run
        * set_key_points - defines key points for geometric model in Ansys
        * set_mesh - generates areas from key points and difines a mesh with pre-defined element size
        * set_boundary_conditions - sets boundary conditions

    """

    def set_parameters(self, material: Material, height=400, width=160, thickness=2,
                       theta=0, alpha=0.5, force=10000,
                       esize=0.5, ref_window=12):
        """Sets variables and parameters for Ansys run.

        Args:
            material: elastic law
            height: of the specimen
            width: of the specimen
            thickness: of the specimen
            theta: crack angle
            alpha: relative crack length (alpha = crack length / width / 2)
            force: traction force on upper boundary
            esize: element size in refined area
            ref_window: radius of squared refinement window around the crack tip

        """
        # Variables
        self.mapdl.run("PI=4.0*atan(1.0)")

        self.mapdl.run(f"h={height}")  # specimen height
        self.mapdl.run(f"w={width}")  # specimen width
        self.mapdl.run(f"t={thickness}")  # specimen thickness

        self.mapdl.run(f"thxy={theta}")  # Crack angle
        self.mapdl.run(f"alpha={alpha}")  # 2a/w

        self.mapdl.run(f"force={force}")  # Kraft

        self.mapdl.run(f"esize={esize}")
        self.mapdl.run(f"ref_window={ref_window}")

        self.mapdl.run("a=alpha*w/2")  # crack length

        # Material properties
        self.mapdl.mp("EX", 1, material.E)  # Young modulus (N/mm^2)
        self.mapdl.mp("NUXY", 1, material.nu_xy)  # Poisson ration

        self.mapdl.et(1, "PLANE182")  # 3-D 8-Node Structural Solid
        if material.plane_strain:
            self.mapdl.keyopt(1, 3, 2)
        else:  # plane stress
            self.mapdl.keyopt(1, 3, 0)

    def set_key_points(self):
        self.mapdl.local(11, 0, "", "", "", "thxy")
        self.mapdl.clocal(12, 0, "a")
        self.mapdl.csys(11)
        self.mapdl.clocal(13, 0, "-a", "", "", 180)
        self.mapdl.csys(0)
        self.mapdl.k(1, 0, 0)
        self.mapdl.csys(11)
        self.mapdl.k(2, "a", 0)
        self.mapdl.csys(0)
        self.mapdl.k(3, "w/2", 0)
        self.mapdl.k(4, "w/2", "h/2")
        self.mapdl.k(5, "w/2*alpha", "h/2")
        self.mapdl.k(6, 0, "h/2")
        self.mapdl.k(7, "w/2", "-h/2")
        self.mapdl.k(8, "w/2*alpha", "-h/2")
        self.mapdl.k(9, 0, "-h/2")
        self.mapdl.k(10, 0, 0)
        self.mapdl.csys(11)
        self.mapdl.k(11, "-a", 0)
        self.mapdl.csys(0)
        self.mapdl.k(12, "-w/2", 0)
        self.mapdl.k(13, "-w/2", "h/2")
        self.mapdl.k(14, "-w/2*alpha", "h/2")
        self.mapdl.k(15, "-w/2", "-h/2")
        self.mapdl.k(16, "-w/2*alpha", "-h/2")

    def set_mesh(self):
        # mesh elements
        self.mapdl.a(1, 2, 5, 6)
        self.mapdl.a(2, 3, 4, 5)
        self.mapdl.a(10, 2, 8, 9)
        self.mapdl.a(2, 3, 7, 8)
        self.mapdl.a(1, 6, 14, 11)
        self.mapdl.a(11, 14, 13, 12)
        self.mapdl.a(10, 9, 16, 11)
        self.mapdl.a(12, 11, 16, 15)

        # mesh refinement
        self.mapdl.wpcsys("", 12)
        self.mapdl.wpoffs("-ref_window")
        self.mapdl.wprota("", "", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 12)
        self.mapdl.wpoffs("ref_window")
        self.mapdl.wprota("", "", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 12)
        self.mapdl.wpoffs("", "-ref_window")
        self.mapdl.wprota("", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 12)
        self.mapdl.wpoffs("", "ref_window")
        self.mapdl.wprota("", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)

        # meshing
        self.mapdl.csys(12)  # Fine mesh around crack tip
        self.mapdl.asel("S", "LOC", "X", "-ref_window", "ref_window")
        self.mapdl.asel("R", "LOC", "Y", "-ref_window", "ref_window")
        self.mapdl.esize("esize")
        self.mapdl.amesh("ALL")
        self.mapdl.asel("INVE")  # Coarse mesh away from crack tip
        self.mapdl.esize(2)
        self.mapdl.amesh("ALL")
        self.mapdl.allsel()
        self.mapdl.csys(0)
        self.mapdl.get("nel", "ELEM", 0, "COUNT")
        self.mapdl.get("nno", "NODE", 0, "COUNT")
        self.mapdl.allsel()

    def set_boundary_conditions(self):
        self.mapdl.nsel("S", "LOC", "Y", "h/2")  # select upper boundary
        self.mapdl.cp("NEXT", "UY", "ALL")  # couple y-displacement
        self.mapdl.nsel("R", "LOC", "X", 0)  # select middle point on upper boundary
        self.mapdl.f("ALL", "FY", "force")  # prescribe traction force
        self.mapdl.allsel()
        self.mapdl.nsel("S", "LOC", "Y", "-h/2")  # select lower boundary
        self.mapdl.d("ALL", "UY", 0)  # no y-displacement on lower boundary
        self.mapdl.allsel()
        self.mapdl.nsel("S", "LOC", "Y", "-h/2")  # select lower boundary
        self.mapdl.d("ALL", "UX", 0)  # no x-displacement on lower boundary
        self.mapdl.nsel("S", "LOC", "Y", "h/2")  # select upper boundary
        self.mapdl.d("ALL", "UX", 0)  # no x-displacemnent on upper boundary
        self.mapdl.allsel()


class CTSimulation(AnsysSimulation):
    """Class for CT Specimen Simulation in ANSYS MECHANICAL.

    Methods:
        * set_parameters - variables and parameters for Ansys run
        * set_key_points - definition of keypoints for specimen geometry
        * set_mesh - generation of areas and meshing with pre-defines element size
        * set_boundary_conditions - definition of boundary conditions

    """

    def set_parameters(self, material: Material, width=75, thickness=10,
                       alpha=0.5, force=10000,
                       esize=0.5, ref_window=8):
        """Sets variables and parameters for Ansys run.

        Args:
            material: elastic law
            width: of the specimen
            thickness: of the specimen
            alpha: relative crack lenght (alpha = crack length / width / 2)
            force: traction force on upper boundary
            esize: element size in refined area
            ref_window: radius of squared refinement window around the crack tip

        """
        # Variables
        self.mapdl.run("PI=4.0*atan(1.0)")

        self.mapdl.run(f"w={width}")  # specimen width
        self.mapdl.run(f"t={thickness}")  # specimen thickness

        self.mapdl.run(f"alpha={alpha}")  # 2a/w

        self.mapdl.run(f"force={force}")  # Kraft

        self.mapdl.run(f"esize={esize}")
        self.mapdl.run(f"ref_window={ref_window}")

        self.mapdl.run("a=alpha*w")  # crack length

        self.mapdl.run("h=1.2*w")  # height of the specimen
        self.mapdl.run("BR=1.0*w")  # width of specimen
        self.mapdl.run("BL=0.25*w")  # width right from hole
        self.mapdl.run("XB=0.0")  # x-coordinate of hole
        self.mapdl.run("YB=0.275*w")  # y-coordinate of hole
        self.mapdl.run("DB=0.25*w")  # diameter of hole

        # Material properties
        self.mapdl.mp("EX", 1, material.E)  # Young modulus (N/mm^2)
        self.mapdl.mp("NUXY", 1, material.nu_xy)  # Poisson ration
        # HOLE
        self.mapdl.mp("EX", 2, 1000000)
        self.mapdl.mp("NUXY", 2, 0.5)

        self.mapdl.et(1, "PLANE182")  # 3-D 8-Node Structural Solid
        self.mapdl.keyopt(1, 3, 3)  # plane stress with thickness input
        self.mapdl.r(1, "t")
        self.mapdl.et(2, "BEAM188")
        self.mapdl.sectype(1, "BEAM", "RECT")
        self.mapdl.secdata(10, 10)

    def set_key_points(self):
        self.mapdl.local(20, 0, "a")
        self.mapdl.csys(0)
        self.mapdl.k(1, "-BL", 0)
        self.mapdl.k(2, "a", 0)
        self.mapdl.k(3, "BR", 0)
        self.mapdl.k(4, "BR", "h/2")
        self.mapdl.k(5, "a", "h/2")
        self.mapdl.k(6, "-BL", "h/2")
        self.mapdl.k(7, "-BL", 0)
        self.mapdl.k(8, "-BL", "-h/2")
        self.mapdl.k(9, "a", "-h/2")
        self.mapdl.k(10, "BR", "-h/2")

    def set_mesh(self):
        # set elements
        self.mapdl.a(1, 2, 5, 6)
        self.mapdl.a(2, 3, 4, 5)
        self.mapdl.a(7, 2, 9, 8)
        self.mapdl.a(2, 3, 10, 9)

        # mesh refinement
        self.mapdl.cyl4("XB", "YB", "DB/2")
        self.mapdl.cyl4("XB", "-YB", "DB/2")
        self.mapdl.asba("ALL", 5)
        self.mapdl.asba("ALL", 6)
        self.mapdl.allsel()
        self.mapdl.wpcsys("", 20)
        self.mapdl.wpoffs("-ref_window")
        self.mapdl.wprota("", "", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 20)
        self.mapdl.wpoffs("ref_window")
        self.mapdl.wprota("", "", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 20)
        self.mapdl.wpoffs("", "-ref_window")
        self.mapdl.wprota("", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)
        self.mapdl.wpcsys("", 20)
        self.mapdl.wpoffs("", "ref_window")
        self.mapdl.wprota("", 90)
        self.mapdl.asbw("ALL", "", "DELETE")
        self.mapdl.wpcsys("", 0)

        # meshing
        self.mapdl.type(1)
        self.mapdl.mat(1)
        self.mapdl.csys(20)  # Fine mesh around crack tip
        self.mapdl.asel("S", "LOC", "X", "-ref_window", "ref_window")
        self.mapdl.asel("R", "LOC", "Y", "-ref_window", "ref_window")
        self.mapdl.esize("esize")
        self.mapdl.amesh("ALL")
        self.mapdl.asel("INVE")  # Coarse mesh away from crack tip
        self.mapdl.esize(2)
        self.mapdl.amesh("ALL")
        self.mapdl.csys(0)
        self.mapdl.allsel()

        # force introduction
        self.mapdl.type(2)
        self.mapdl.mat(2)
        self.mapdl.get("Nmax", "NODE", 0, "num", "maxd")
        self.mapdl.n("Nmax+1", "XB", "YB")
        self.mapdl.n("Nmax+2", "XB", "-YB")
        self.mapdl.local(21, 1, "XB", "YB")
        self.mapdl.csys(21)
        self.mapdl.run("angle=180")
        self.mapdl.nsel("S", "LOC", "X", "DB/2")
        self.mapdl.nsel("R", "LOC", "Y", "90-angle", "90+angle")
        self.mapdl.get("nsel", "NODE", 0, "COUNT")
        self.mapdl.run("nodenum=0")
        with self.mapdl.non_interactive:
            self.mapdl.run("*DO,kk,1,nsel")
            self.mapdl.run("nodenum=ndnext(nodenum)")
            self.mapdl.e("nodenum", "Nmax+1")
            self.mapdl.run("*ENDDO")
        self.mapdl.local(22, 1, "XB", "-YB")
        self.mapdl.csys(22)
        self.mapdl.nsel("S", "LOC", "X", "DB/2")
        self.mapdl.nsel("R", "LOC", "Y", "270-angle", "270+angle")
        self.mapdl.get("nsel", "NODE", 0, "COUNT")
        self.mapdl.run("nodenum=0")
        with self.mapdl.non_interactive:
            self.mapdl.run("*DO,kk,1,nsel")
            self.mapdl.run("nodenum=ndnext(nodenum)")
            self.mapdl.e("nodenum", "Nmax+2")
            self.mapdl.run("*ENDDO")
        self.mapdl.csys(0)

    def set_boundary_conditions(self):
        self.mapdl.csys(0)
        self.mapdl.seltol(0.001)
        # top
        self.mapdl.csys(21)
        self.mapdl.nsel("S", "LOC", "X", 0)
        self.mapdl.nsel("R", "LOC", "Y", 0)
        self.mapdl.f("ALL", "FY", "Force")
        self.mapdl.d("ALL", "UX")
        self.mapdl.d("ALL", "UZ")
        self.mapdl.d("ALL", "ROTX")
        self.mapdl.d("ALL", "ROTY")
        # bottom
        self.mapdl.csys(22)
        self.mapdl.nsel("S", "LOC", "X", 0)
        self.mapdl.nsel("R", "LOC", "Y", 0)
        self.mapdl.d("ALL", "UX")
        self.mapdl.d("ALL", "UY")
        self.mapdl.d("ALL", "UZ")
        self.mapdl.d("ALL", "ROTX")
        self.mapdl.d("ALL", "ROTY")
        self.mapdl.csys(0)
        self.mapdl.allsel()

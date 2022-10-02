"""Generate samples from 6D distributions."""
import numpy as np


class PhaseSpaceGen2D:
    """Sample from 2D distribution.

    The 2D distribution is read from a file with the following format:
        % comment line
        0.0       x1       x2       ...   x_nx
        xp1    val_1_1   val_1_2    ... val_1_nx
        xp2    val_2_1   val_2_2    ... val_2_nx
        ...    ...       ...        ... ...
        xp_nxp val_nxp_1 val_nxp_2  ... val_nxp_nx
    
    The coordinates are assumed to be mm and mrad. The array values should
    be normalized to the range [0, 1].
    
    To-do: rewrite with numpy.
    """
    def __init__(self, filename, x_max=1.0e36, xp_max=1.0e36, threshold=3e-4):
        ext = filename.split('.')[-1]
        if ext == ".txt":
            delimiter = " "
        elif ext == ".csv":
            delimiter = ","
        else:
            raise ValueError('Unknown extension.')
            
        self.pdf = []
        arr_in = np.genfromtxt(filename, comments="%", 
                               delimiter=delimiter, filling_values=0.0)

        # The [0, 0] component of `arr_in` is a linear correlation that
        # should be inserted into the bunch.
        self.slope = arr_in[0, 0]

        # Parse info from data file; get x endpoints.
        res_arr = arr_in[0, 1:]
        self.nx = len(res_arr) - 1
        endpoints = np.array([float(res_arr[1]), float(res_arr[-1])])
        self.x_min = np.min(endpoints)
        self.x_max = np.max(endpoints)
        self.x_step = (self.x_max - self.x_min) / (self.nx - 1)

        xp_arr = arr_in[1:, 0]
        self.val_matrix = arr_in[1:, 1:]

        # Threshold
        val_peak = self.val_matrix.max().max()
        self.val_matrix[self.val_matrix < val_peak * threshold] = 0.0
        # -- re-normalize to sum
        self.val_matrix /= self.val_matrix.sum().sum()
        # -- get x' endpoints
        self.nxp = len(xp_arr)
        self.xp_min = xp_arr[0]
        self.xp_max = xp_arr[len(xp_arr) - 1]
        self.xp_step = (self.xp_max - self.xp_min) / (self.nxp - 1)
        # print("debug x_min=", self.x_min, " x_max=", self.x_max,
        #       " xp_min=", self.xp_min, " xp_max=", self.xp_max)
        # print("debug xp_step=", self.xp_step, " nx=", self.nx,
        #       " nxp=", self.nxp, " lenMtrY=", len(val_matrix), " lenMtrX=", len(val_matrix[0]))
        # --------------------------------------------------------------
        # -- make 2D grid object for PDF
        self.grid2D = Grid2D(
            self.nx, self.nxp, self.x_min, self.x_max, self.xp_min, self.xp_max
        )
        self.x_min_gen = self.x_min
        self.xp_min_gen = self.xp_min
        if self.x_min_gen < -math.fabs(x_max):
            self.x_min_gen = -math.fabs(x_max)
        if self.xp_min_gen < -math.fabs(xp_max):
            self.xp_min_gen = -math.fabs(xp_max)
        self.x_max_gen = self.x_max
        self.xp_max_gen = self.xp_max
        if self.x_max_gen > math.fabs(x_max):
            self.x_max_gen = math.fabs(x_max)
        if self.xp_max_gen > -math.fabs(xp_max):
            self.xp_max_gen = math.fabs(xp_max)
        # --------------------------------------------------------------
        # -- deposit values in PDF grid
        for ix in range(self.nx):
            for ixp in range(self.nxp):
                val = self.val_matrix[ixp][ix]
                # -- assign value to 2D grid
                self.grid2D.setValue(val, ix, ixp)
        # --------------------------------------------------------------
        # -- make 2D grid for CDF
        self.int_grid2D = Grid2D(
            self.nx, self.nxp, self.x_min, self.x_max, self.xp_min, self.xp_max
        )
        for ix in range(self.nx):
            self.int_grid2D.setValue(0.0, ix, 0)
            for ixp in range(1, self.nxp):
                val_0 = self.int_grid2D.getValueOnGrid(ix, ixp - 1)
                val = (
                    self.grid2D.getValueOnGrid(ix, ixp - 1)
                    + self.grid2D.getValueOnGrid(ix, ixp)
                ) / 2.0
                self.int_grid2D.setValue(val + val_0, ix, ixp)
        self.s_int_arr = [
            0.0,
        ]
        for ix in range(1, self.nx):
            val_0 = self.s_int_arr[ix - 1]
            val = (
                self.int_grid2D.getValueOnGrid(ix - 1, self.nxp - 1)
                + self.int_grid2D.getValueOnGrid(ix, self.nxp - 1)
            ) / 2.0
            self.s_int_arr.append(val + val_0)
        s_tot = self.s_int_arr[self.nx - 1]
        if s_tot > 0.0:
            for ix in range(self.nx):
                self.s_int_arr[ix] = self.s_int_arr[ix] / s_tot
        for ix in range(self.nx):
            s_tot = self.int_grid2D.getValueOnGrid(ix, self.nxp - 1)
            if s_tot > 0.0:
                for ixp in range(self.nxp):
                    val = self.int_grid2D.getValueOnGrid(ix, ixp) / s_tot
                    self.int_grid2D.setValue(val, ix, ixp)
        # print("debug =================================gen created =============")

    def get_x(self):
        g = random.random()
        ind_x_0 = 0
        ind_x_1 = self.nx - 1
        count_max = 200
        count = 0
        val0 = self.s_int_arr[ind_x_0]
        val1 = self.s_int_arr[ind_x_1]
        while not abs(ind_x_1 - ind_x_0) <= 1:
            count += 1
            if count > count_max:
                print("debug problem with X generation g=", g)
                print("debug val0=", val0, " val1=", val1)
                print("debug ind_xp_0=", ind_xp_0, " ind_xp_1=", ind_xp_1)
                print("debug s_int_arr=", self.s_int_arr)
                sys.exit(1)
            if g > val0 and g <= val1:
                ind_new = int((ind_x_0 + ind_x_1) / 2)
                if g > self.s_int_arr[ind_new]:
                    ind_x_0 = ind_new
                    val0 = self.s_int_arr[ind_x_0]
                else:
                    ind_x_1 = ind_new
                    val1 = self.s_int_arr[ind_x_1]
        coeff = (g - val0) / (val1 - val0)
        x = self.x_min + (ind_x_0 + coeff) * self.x_step
        return (x, coeff, ind_x_0, ind_x_1)

    def get_xp(self, coeff, ind_x_0, ind_x_1):
        g = random.random()
        ind_x = ind_x_0
        if g > coeff:
            ind_x = ind_x_1
        g = random.random()
        ind_xp_0 = 0
        ind_xp_1 = self.nxp - 1
        count_max = 200
        count = 0
        val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
        val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        if val1 == 0.0:
            if ind_x == ind_x_0:
                ind_x = ind_x_1
            else:
                ind_x = ind_x_0
        val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
        val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        while not abs(ind_xp_1 - ind_xp_0) <= 1:
            count += 1
            if count > count_max:
                print("debug problem with XP generation g=", g)
                print("debug ind_x=", ind_x)
                print("debug val0=", val0, " val1=", val1)
                print("debug coeff=", coeff)
                print("debug  ind_x_0=", ind_x_0, "   ind_x_1=", ind_x_1)
                print("debug ind_xp_0=", ind_xp_0, " ind_xp_1=", ind_xp_1)
                print("debug s_int_arr[ind_x_0]=", self.s_int_arr[ind_x_0])
                print("debug s_int_arr[ind_x_1]=", self.s_int_arr[ind_x_1])
                print("debug s_int_arr=", self.s_int_arr)
                for ixp in range(self.nxp):
                    print(
                        "debug ix=",
                        ind_x_0,
                        " ixp=",
                        ixp,
                        " val0=",
                        self.int_grid2D.getValueOnGrid(ind_x_0, ixp),
                        " val1=",
                        self.int_grid2D.getValueOnGrid(ind_x_1, ixp),
                    )
                sys.exit(1)
            if g > val0 and g <= val1:
                ind_new = int((ind_xp_0 + ind_xp_1) / 2)
                if g > self.int_grid2D.getValueOnGrid(ind_x, ind_new):
                    ind_xp_0 = ind_new
                    val0 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_0)
                else:
                    ind_xp_1 = ind_new
                    val1 = self.int_grid2D.getValueOnGrid(ind_x, ind_xp_1)
        coeff = (g - val0) / (val1 - val0)
        xp = self.xp_min + (ind_xp_0 + coeff) * self.xp_step
        return xp

    def get_x_xp(self):
        count, count_max = 0, 1000
        while True:
            (x, coeff, ind_x_0, ind_x_1) = self.get_x()
            xp = self.get_xp(coeff, ind_x_0, ind_x_1)

            if abs(x) < self.x_max_gen and abs(xp) < self.xp_max_gen:
                # -- add in linear correlation
                x += xp * self.slope
                return (x / 1000.0, xp / 1000.0)
            count += 1
            if count > count_max:
                print("debug problem with X and XP generation count=", count)
                print("debug x=", x, " xp=", xp)
                print("debug self.x_max_gen = {}".format(self.x_max_gen))
                print("debug self.xp_max_gen = {}".format(self.xp_max_gen))
                sys.exit(1)

    def get_y_yp(self):
        return self.get_x_xp()

    def get_z_zp(self):
        return self.get_x_xp()

    def gen_pdf(self):
        """Return nrarray from Grid2D."""
        self.pdf = self.val_matrix / self.val_matrix.sum()

    def grid_sample(self, n_parts=0):
        # make number density array out of PDF grid
        # returns arrays of x and x' coordinates
        #

        # -- make number density grid based on PDF
        Ndistr = np.floor(self.pdf * n_parts).astype(int)
        n_parts = Ndistr.sum()

        # -- define coordinate grid
        xaxis = np.linspace(self.x_min, self.x_max, num=self.nx)
        xpaxis = np.linspace(self.xp_min, self.xp_max, num=self.nxp)
        [XG, XPG] = np.meshgrid(xaxis, xpaxis)

        # -- loop through grid + deposit Ndistr(i,j) particles at each point
        # could be combined into 1 loop (do x and y simultaneously),
        # but left general in case x and y scans have different resolutions
        X, XP = np.zeros([2, n_parts])
        counter = 0
        for i in range(np.shape(Ndistr)[0]):
            for j in range(np.shape(Ndistr)[1]):
                if Ndistr[i, j] > 0:
                    counterend = counter + Ndistr[i, j]
                    X[counter:counterend] = XG[i, j]
                    XP[counter:counterend] = XPG[i, j]
                    counter = counterend

        # -- spread particles out from gridpoint locations through a uniform kernel
        # scale from mm, mrad to meters, rad
        # right now width of each kernel is twice grid spacing, is this intentional?
        x = (X + (np.random.rand(1, n_parts) - 0.5) * 2 * self.x_step) / 1000.0
        xp = (XP + (np.random.rand(1, n_parts) - 0.5) * 2 * self.xp_step) / 1000.0

        # -- random sampling to remove correlations in x-y, xp-yp
        xind = np.random.permutation(range(np.shape(x)[1]))

        # -- return shuffled x,x' distribution
        return (x[0, xind], xp[0, xind])


class PhaseSpaceGenZPartial:
    """Generates (z, dE) distribution using 1D e-profile.

    Assumes Gaussian phase to fit specified emittance and beta function.
    """
    def __init__(
        self, 
        filename, 
        twiss_z, 
        zdistributor=GaussDist1D, 
        cut_off=-1, 
        threshold=1e-3,
    ):
        emitZ = twiss_z.emittance
        betaZ = twiss_z.beta
        alphaZ = twiss_z.alpha

        print("========= Input Twiss ===========")
        print(
            "alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "
            % (alphaZ, betaZ, emitZ * 1.0e6)
        )

        ## -- load input dE distribution
        deinput = np.loadtxt(filename)
        deval = deinput[:, 0] * 1e-3  # units GeV

        # -- input file is already thresholded and normalized (sum=1)
        pdf = deinput[:, 1]

        # -- calculate RMS alpha to preserve emittance and beta
        self.dErms = np.sqrt(np.sum(pdf * deval**2) / np.sum(pdf))
        alphaZ = -np.sqrt(betaZ * self.dErms**2 / emitZ - 1)

        # -- construct cumulative distribution function
        dEstep = (
            np.mean(deval[1:] - deval[0:-1]) * 1e3
        )  # this unit has to be in MeV to give correct cdf
        cdf = np.zeros(len(pdf))
        for i in range(1, len(cdf)):
            cdf[i] = np.sum(pdf[0:i]) * dEstep

        # -- function that returns dE [GeV] given index in range [0,1]
        self.invcdf = interp1d(cdf, deval, bounds_error=False, fill_value="extrapolate")

        print("========= Adjusted Twiss ===========")
        print(
            "alpha beta emitt[mm*MeV] Z= %6.4f %6.4f %6.4f "
            % (alphaZ, betaZ, emitZ * 1.0e6)
        )

        twiss_z = TwissContainer(alphaZ, betaZ, emitZ)

        # -- distributor for 2D Gaussian distribution
        self.distributor = None
        if zdistributor == WaterBagDist1D:
            self.distributor = zdistributor(twiss_z)
        else:
            self.distributor = zdistributor(twiss_z, cut_off)

    def get_z_zp(self):
        """Sample from z-dE distribution."""
        (z, dE) = self.distributor.getCoordinates()
        ind01 = norm.cdf(
            dE, scale=self.dErms
        )  # index of particle from [0,1] distribution
        dE = self.invcdf(ind01)
        return (z, dE)

    def gen_pdf(self):
        print("Method 'gen_pdf' not enabled")

    def grid_sample(self, n_parts=0):
        print("Method 'grid_sample' not enabled")
        return (0, 0)
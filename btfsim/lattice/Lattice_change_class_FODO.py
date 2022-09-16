import sys
import time
import math
import numpy as np
import os
import os.path
import btfsim.util.Defaults as default


class new_lattice:
    def __init__(self, Lat_fileName, lattice_len):

        lattice_len = float(lattice_len)

        Cal_stop = lattice_len
        # print "=============================%%%%%%%%%%%%%%%%%%",Cal_stop

        if 1.06039 < Cal_stop < 1.22239:  # Space between matching magnets and FODO
            line_num = 15
            Mag_Num = 4

        else:
            if 5.04139 < Cal_stop < 5.26139:  # FODO end
                line_num = 72
                Mag_Num = 4
            else:
                if 5.40739 < Cal_stop < 5.46739:  # Between MQ1 and MQ2
                    line_num = 75
                    Mag_Num = 5
                else:
                    if 5.61339 < Cal_stop < 6.94189:  # Between MQ2 and last Bend
                        line_num = 78
                        Mag_Num = 6
                    else:
                        line_num = 81
                        Mag_Num = 6

                        # else:
                        # 	print "Please Choose another position"
                        # 	line_num = 45
                        # 	Mag_Num = 9
                        # 	Cal_stop = 5.120950348

        # print line_num
        self.Magnumber = Mag_Num

        # f1 = open('btf_mebt_BTF.xml','r+')
        f1 = open(Lat_fileName, "r+")
        defaults = default.getDefaults()
        f2 = open(defaults.latticedir + "temp_btf_mebt.xml", "w+")
        infos = f1.readlines()
        f1.seek(0, 0)

        for i in range(line_num):
            # if (line_num !=45 or (line_num ==45 and nnn ==0)):
            if i == 2:
                infos[i] = infos[i].replace("7.500465174", str(Cal_stop))
            f2.write(infos[i])
        for j in range(81, 88, 1):
            f2.write(infos[j])

        f1.close()
        f2.close()

    def MagNum(self):
        return self.Magnumber

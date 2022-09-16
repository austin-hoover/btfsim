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

        if 0.161 < Cal_stop < 0.281:
            line_num = 6
            Mag_Num = 1

        else:
            if 0.347 < Cal_stop < 0.527:
                line_num = 9
                Mag_Num = 2
            else:
                if 0.623 < Cal_stop < 0.723:
                    line_num = 12
                    Mag_Num = 3
                else:
                    if 0.819 < Cal_stop < 1.626:
                        line_num = 15
                        Mag_Num = 4
                    else:
                        if 1.722 < Cal_stop < 1.835:
                            line_num = 21
                            Mag_Num = 5
                        else:
                            if 1.931 < Cal_stop < 2.9105:
                                line_num = 24
                                Mag_Num = 6
                            else:
                                if 3.469075174 < Cal_stop < 3.607075174:
                                    line_num = 33
                                    Mag_Num = 6

                                else:
                                    if 3.703075174 < Cal_stop < 3.853075174:
                                        line_num = 36
                                        Mag_Num = 7
                                    else:
                                        if 3.949075174 < Cal_stop < 4.299075174:
                                            line_num = 39
                                            Mag_Num = 8
                                        else:
                                            if 4.395075174 < Cal_stop < 4.533075174:
                                                line_num = 42
                                                Mag_Num = 9
                                            else:
                                                # if (5.091650348< Cal_stop <= 5.120950348):
                                                line_num = 45
                                                Mag_Num = 9

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
                infos[i] = infos[i].replace("5.120950348", str(Cal_stop))
            f2.write(infos[i])
        for j in range(45, 52, 1):
            f2.write(infos[j])

        f1.close()
        f2.close()

    def MagNum(self):
        return self.Magnumber

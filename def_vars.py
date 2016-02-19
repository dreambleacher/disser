#!/usr/bin/env python
# -*- coding: utf-8 -*-


mod_coef=[
u'FLaSG1_CfResL',
u'FLaSG2_CfResL',
u'FLaSG3_CfResL',
u'FLaSG4_CfResL',
u'YD11D01_2_Hnom',
u'YD12D01_2_Hnom',
u'YD13D01_2_Hnom',
u'YD14D01_2_Hnom',
#u'YHSIEVE_TUN(1)',
u'Loop_CfResL1',
u'Loop_CfResL2',
u'Loop_CfResL3',
u'Loop_CfResL4',
u'SG_CfResL1',
u'SG_CfResL2',
u'SG_CfResL3',
u'SG_CfResL4',
u'YhqCor1_eqf',
u'YhqCor2_eqf',
u'YhqCor3_eqf'
]

mod_coef_delta={
u'FLaSG1_CfResL':[0.,2.],
u'FLaSG2_CfResL':[0.,2.],
u'FLaSG3_CfResL':[0.,2.],
u'FLaSG4_CfResL':[0.,2.],
u'YD11D01_2_Hnom':[70,110],
u'YD12D01_2_Hnom':[70,110],
u'YD13D01_2_Hnom':[70,110],
u'YD14D01_2_Hnom':[70,110],
#u'YHSIEVE_TUN(1)':[,],
u'Loop_CfResL1':[1.,2.],
u'Loop_CfResL2':[1.,2.],
u'Loop_CfResL3':[1.,2.],
u'Loop_CfResL4':[1.,2.],
u'SG_CfResL1':[1.,2.],
u'SG_CfResL2':[1.,2.],
u'SG_CfResL3':[1.,2.],
u'SG_CfResL4':[1.,2.],
u'YhqCor1_eqf':[0,5],
u'YhqCor2_eqf':[0,5],
u'YhqCor3_eqf':[0,5]
}



yexpvar=[
'Tgor1',
'Tgor2',
'Tgor3',
'Tgor4',
'Thol1',
'Thol2',
'Thol3',
'Thol4',
'dPgcn1',
'dPgcn2',
'dPgcn3',
'dPgcn4',
'Pzone1',
##'Pzone2',
##'Ntep=0.,Naz=0.,Nrr=0.,N1k=0.,N2k=0.,Naknp=0.,Nturb=0.,
'Tpv1',
'Tpv2',
'Tpv3',
'Tpv4',
'Gpv1',
'Gpv2',
'Gpv3',
'Gpv4',
'Ppg1',
'Ppg2',
'Ppg3',
'Ppg4'
]


deriv_test ={ #погрешности нормировки !только чтобы отсечь лишнее
u'Gpv1':1,
u'Gpv2':1,
u'Gpv3':1,
u'Gpv4':1,
u'N1k':0.01,
u'N2k':0.01,
u'Naknp':0.01,
u'Naz':0.01,
u'Nrr':0.01,
u'Ntep':0.01,
u'Nturb':0.01,
u'Pgpk':0.1,
u'Ppg1':0.1,
u'Ppg2':0.1,
u'Ppg3':0.1,
u'Ppg4':0.1,
u'Pzone1':0.1,
u'Pzone2':0.1,
u'Tgor1':0.1,
u'Tgor2':0.1,
u'Tgor3':0.1,
u'Tgor4':0.1,
u'Thol1':0.1,
u'Thol2':0.1,
u'Thol3':0.1,
u'Thol4':0.1,
u'Tpv1':0.1,
u'Tpv2':0.1,
u'Tpv3':0.1,
u'Tpv4':0.1,
u'dPgcn1':0.1,
u'dPgcn2':0.1,
u'dPgcn3':0.1,
u'dPgcn4':0.1,
u'gkgtn':1,
u'gpvd1':1,
u'gpvd2':1,
u'ppv1':0.1,
u'ppv2':0.1,
u'ppv3':0.1,
u'ppv4':0.1,
u'ppvd1':0.1,
u'ppvd2':0.1,
u'tpvd1':0.1,
u'tpvd2':0.1}



def main():
    pass

if __name__ == '__main__':
    main()

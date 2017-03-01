#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

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
u'rot_coef',
#u'YhqCor1_eqf',
#u'YhqCor2_eqf',
#u'YhqCor3_eqf',
u'Nin', u'Pazin', u'Pgpkin',u'Tpvdain', u'Tpvdbin'
]

mod_coef_lable={
u'FLaSG1_CfResL':ur'$\xi_{пг1}$',
u'FLaSG2_CfResL':ur'$\xi_{пг2}$',
u'FLaSG3_CfResL':ur'$\xi_{пг3}$',
u'FLaSG4_CfResL':ur'$\xi_{пг4}$',
u'YD11D01_2_Hnom':ur'$k_{Gгцн1}$',
u'YD12D01_2_Hnom':ur'$k_{Gгцн2}$',
u'YD13D01_2_Hnom':ur'$k_{Gгцн3}$',
u'YD14D01_2_Hnom':ur'$k_{Gгцн4}$',
#u'YHSIEVE_TUN(1)',
u'Loop_CfResL1':ur'$\xi_{гцк1}$',
u'Loop_CfResL2':ur'$\xi_{гцк2}$',
u'Loop_CfResL3':ur'$\xi_{гцк3}$',
u'Loop_CfResL4':ur'$\xi_{гцк4}$',
u'SG_CfResL1':ur'$\xi_{тр.пг1}$',
u'SG_CfResL2':ur'$\xi_{тр.пг2}$',
u'SG_CfResL3':ur'$\xi_{тр.пг3}$',
u'SG_CfResL4':ur'$\xi_{тр.пг4}$',
u'rot_coef':ur'$k_{закр}$',
u'YhqCor1_eqf':ur'$k_{смеш1}$',
u'YhqCor2_eqf':ur'$k_{смеш2}$',
u'YhqCor3_eqf':ur'$k_{смеш3}$',
u'Nin':ur'$N_{АЗ}$',
u'Pazin':ur'$P_{АЗ}$',
u'Pgpkin':ur'$P_{гпк}$',
u'Tpvdain':ur'$t_{пвд1}$',
u'Tpvdbin':ur'$t_{пвд2}$'
}

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
u'Loop_CfResL1':[0.1,1.5],
u'Loop_CfResL2':[0.1,1.5],
u'Loop_CfResL3':[0.1,1.5],
u'Loop_CfResL4':[0.1,1.5],
u'SG_CfResL1':[1.,2.5],
u'SG_CfResL2':[1.,2.5],
u'SG_CfResL3':[1.,2.5],
u'SG_CfResL4':[1.,2.5],
u'rot_coef':[-5,5],
#u'YhqCor1_eqf':[0.,5.],
#u'YhqCor2_eqf':[0.,5.],
#u'YhqCor3_eqf':[0.,5.],
u'Nin':[94.,108.],
u'Pgpkin':[59.5,61.7],
u'Pazin':[158.5,159.9],
u'Tpvdain':[200.,220.],
u'Tpvdbin':[200.,220.]
}

mod_coef_delta_m=np.array([mod_coef_delta[p] for p in mod_coef])

vh_param=[
'N',
'Pgpk',
'tpvd1',
'tpvd2',
'Paz',
]


vh_delt=dict(
N=[96.,103.],
Pgpk=[59.5,61.5],
Paz=[158.5,159.9],
tpvd1=[200.,220.],
tpvd2=[200.,220.])

yexpvar=[
'Tgor1',#0
'Tgor2',
'Tgor3',
'Tgor4',
'Thol1',#4
'Thol2',
'Thol3',
'Thol4',
'dPgcn1',
'dPgcn2',
'dPgcn3',#10
'dPgcn4',
'Pzone1',
'Pzone2',#13
'Ntep','Naz','Nrr','N1k','N2k','Naknp','Nturb',#20
'Tpv1',#21
'Tpv2',#
'Tpv3',
'Tpv4',
'Gpv1',#25
'Gpv2',
'Gpv3',
'Gpv4',
'Ppg1',#29
'Ppg2',
'Ppg3',
'Ppg4',
'Pgpk',
'tpvd1','tpvd2',
'ppvd1','ppvd2',
'gpvd1','gpvd2',
'ppv1','ppv2','ppv3','ppv4',
'gkgtn',
'Gp1kgs','Gp2kgs','Gp3kgs','Gp4kgs'
]

yexpvar_lable={
'Tgor1':ur'$t_{гор1},\ ^\circ C$',
'Tgor2':ur'$t_{гор2},\ ^\circ C$',
'Tgor3':ur'$t_{гор3},\ ^\circ C$',
'Tgor4':ur'$t_{гор4},\ ^\circ C$',
'Thol1':ur'$t_{хол1},\ ^\circ C$',
'Thol2':ur'$t_{хол2},\ ^\circ C$',
'Thol3':ur'$t_{хол3},\ ^\circ C$',
'Thol4':ur'$t_{хол4},\ ^\circ C$',
'dPgcn1':ur'$dP_{гцн1},\ МПа$',
'dPgcn2':ur'$dP_{гцн2},\ МПа$',
'dPgcn3':ur'$dP_{гцн3},\ МПа$',
'dPgcn4':ur'$dP_{гцн4},\ МПа$',
'Pzone1':ur'$P_{АЗ},\ МПа$',
'Pzone2':ur'$dP_{АЗ},\ МПа$',
'Ntep':'$Ntep$','Naz':'$Naz$','Nrr':'$Nrr$','N1k':'$N1k$','N2k':'$N2k$','Naknp':'$Naknp$','Nturb':'$Nturb$',
'Tpv1':ur'$t_{пв1},\ ^\circ C$',
'Tpv2':ur'$t_{пв2},\ ^\circ C$',
'Tpv3':ur'$t_{пв3},\ ^\circ C$',
'Tpv4':ur'$t_{пв4},\ ^\circ C$',
'Gpv1':ur'$G_{пв1},\ \frac{кг}{м^3}$',
'Gpv2':ur'$G_{пв2},\ \frac{кг}{м^3}$',
'Gpv3':ur'$G_{пв3},\ \frac{кг}{м^3}$',
'Gpv4':ur'$G_{пв4},\ \frac{кг}{м^3}$',
'Gp1kgs':ur'$G_{петл1},\ \frac{кг}{с}$',
'Gp2kgs':ur'$G_{петл2},\ \frac{кг}{с}$',
'Gp3kgs':ur'$G_{петл3},\ \frac{кг}{с}$',
'Gp4kgs':ur'$G_{петл4},\ \frac{кг}{с}$',
'Ppg1':ur'$P_{пг1},\ МПа$',
'Ppg2':ur'$P_{пг2},\ МПа$',
'Ppg3':ur'$P_{пг3},\ МПа$',
'Ppg4':ur'$P_{пг4},\ МПа$',
'Pgpk':ur'$P_{гпк},\ МПа$',
'tpvd1':ur'$t_{пвд1},\ ^\circ C$',
'tpvd2':ur'$t_{пвд2},\ ^\circ C$',
'ppvd1':ur'$P_{пвд1},\ МПа$',
'ppvd2':ur'$P_{пвд2},\ МПа$',
'gpvd1':ur'$G_{пвд1},\ \frac{кг}{м^3}$',
'gpvd2':ur'$G_{пвд2},\ \frac{кг}{м^3}$',
'ppv1':ur'$P_{пв1},\ МПа$',
'ppv2':ur'$P_{пв2},\ МПа$',
'ppv3':ur'$P_{пв3},\ МПа$',
'ppv4':ur'$P_{пв4},\ МПа$',
'gkgtn':ur'$G_{кгтн},\ \frac{кг}{м^3}$'
}

deriv_test ={ #погрешности нормировки !только чтобы отсечь лишнее
u'Gp1kgs':1,
u'Gp2kgs':1,
u'Gp3kgs':1,
u'Gp4kgs':1,
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

arch_var_deviation=dict(
    Tgor1=3.,Tgor2=3.,Tgor3=3.,Tgor4=3.,
    Thol1=3.,Thol2=3.,Thol3=3.,Thol4=3.,
    dPgcn1=10000000000,dPgcn2=10000000000,dPgcn3=10000000000,dPgcn4=10000000000,#dPgcn1=0.005*10.197,dPgcn2=0.005*10.197,dPgcn3=0.005*10.197,dPgcn4=0.005*10.197,
    Gp1kgs=0.025*4500,Gp2kgs=0.025*4500,Gp3kgs=0.025*4500,Gp4kgs=0.025*4500,#?? tochnee poschitat bilo - 0.025*4500 new for test
    Pzone1=0.11*10.197,Pzone2=10000000000., #ot baldi max 100
    Ntep=100000000.,Naz=100000000.,Nrr=100000000.,N1k=100000000.,N2k=100000000.,Naknp=100000000.,Nturb=100000000., #ot baldi max 100,N1k=40.
    Tpv1=3.,Tpv2=3.,Tpv3=3.,Tpv4=3.,#ot baldi
    Gpv1=97.,Gpv2=97.,Gpv3=97.,Gpv4=97.,
    Ppg1=0.07*10.197,Ppg2=0.07*10.197,Ppg3=0.07*10.197,Ppg4=0.07*10.197,
    Pgpk=0.01*10.197,#ot baldi
    tpvd1=1.5,tpvd2=1.5,#ot baldi
    ppvd1=0.3*10.197,ppvd2=0.3*10.197,#ot baldi
    gpvd1=103.,gpvd2=103.,#ot baldi
    ppv1=0.3*10.197,ppv2=0.3*10.197,ppv3=0.3*10.197,ppv4=0.3*10.197,#ot baldi
    gkgtn=1000000000.) #ot baldi

def main():
    pass

if __name__ == '__main__':
    main()

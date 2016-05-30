#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        deviation b3
# Purpose: расчет погрешностей переменных полученных с блока 3 клнаэс
#
# Author:      iliyas
#
# Created:     04.04.2016
# Copyright:   (c) iliyas 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib import rc
from freesteam import *
import locale
import os

locale.setlocale(locale.LC_ALL,'rus')
rc('font', **{'family': 'verdana'})
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

try:
    if os.environ['COMPUTERNAME']=='MOLEV':
        dir2='D:/work_place/serg/#b3-svbu/' #work
    else:
        dir2='g:/#work/#b3-svbu/' #home
    #dir2='g:/#work/#b3-svbu/'
    store = pd.HDFStore(dir2+'store.h5')
except IOError:
    print u'Нет доступа к директории ',dir2

datab3=store['alldata']
datab3['N1k'].plot()
plt.show()

datab3['N1k'].index[0]
dt=[]
for i,ind in enumerate(datab3.index):
    dd=datab3.index[i+1]-datab3.index[i]
    dt.append(dd.seconds)

dtser=pd.Series(dt)
indspan=dtser.index[dtser>61]+1

def par_arch_std(par='N1k',pltshow=False):
    u"""
    Функция поиска погрешностей разбитых на отрезки параметров архива
    """
    ib=0
    parstd=[]
    for ii in indspan[:-2]:
        #print ii,datab3['N1k'][ib:ii].std()
        parstd.append(datab3[par][ib:ii].std())
        ib=ii
    serparstd=pd.Series(parstd)
    print '\t',par
    #print serparstd.describe()
    print par,' 0  - ',datab3[par][0]
    print 'std %  - ',serparstd.mean()*100./datab3[par][0]
    print '________'
    if pltshow:
        plt.hist(parstd,bins=20)
        plt.xlabel(yexpvar_lable[u"N1k"]+ur'$, МВт$',fontsize=16)
        plt.show()

for param in datab3.columns:
    #погрешности всех параметров
    par_arch_std(param)

doc_var_deviation=dict(
    Tgor1=2.9,Tgor2=2.9,Tgor3=2.9,Tgor4=2.9,
    Thol1=2.9,Thol2=2.9,Thol3=2.9,Thol4=2.9,
    dPgcn1=0.005*10.197,dPgcn2=0.005*10.197,dPgcn3=0.005*10.197,dPgcn4=0.005*10.197,
    Pzone1=0.11*10.197,Pzone2=10000000000., #ot baldi max 100 po doc 0.004*10.197
    Ntep=100000000.,Naz=100000000.,Nrr=100000000.,N1k=100000000.,N2k=100000000.,Naknp=100000000.,Nturb=40., #ot baldi max 100
    Tpv1=2.2,Tpv2=2.2,Tpv3=2.2,Tpv4=2.2,
    Gpv1=31.,Gpv2=31.,Gpv3=31.,Gpv4=31.,
    Ppg1=0.07*10.197,Ppg2=0.07*10.197,Ppg3=0.07*10.197,Ppg4=0.07*10.197,
    Pgpk=0.01*10.197,#ot baldi
    tpvd1=2.2,tpvd2=2.2,#ot baldi
    ppvd1=0.07*10.197,ppvd2=0.07*10.197,#ot baldi
    gpvd1=97.,gpvd2=97.,
    ppv1=0.11*10.197,ppv2=0.11*10.197,ppv3=0.11*10.197,ppv4=0.11*10.197,
    gkgtn=1000000000.) #ot baldi

#термопары 1-й горячей петли
print datab3[['tgorp1_sr','tgorp1_s5dtc','tgorp1_s4da','tgorp1_s4db','tgorp1_s4dc','tgorp1_s3da','tgorp1_s3db','tgorp1_s3dc','tgorp1_s2da','tgorp1_s2db','tgorp1_s1da','tgorp1_s1db','tgorp1_s1dc']].ix[0]

def calcN1(tgorp1_sr=318.,tholp1_sr=287.,preak=15.6,fpitgcn1=50.,ppgcn1=0.58,nedgcn1=4718.):
    """
    Расчет погрешности мощности по 1 контуру
    """
    #tgorp1_sr=318. #C
    #tholp1_sr=287. #C
    #preak=15.6     #Mpa
    #fpitgcn1=50.
    #ppgcn1=0.58 #Mpa
    #nedgcn1=4718.
    #энтальпии+плотности
    #единичные
    hgor1=steam_pT(preak*10**6,tgorp1_sr+273.15).h
    rogor1=steam_pT(preak*10**6,tgorp1_sr+273.15).rho
    hhol1=steam_pT(preak*10**6,tholp1_sr+273.15).h
    rohol1=steam_pT(preak*10**6,tholp1_sr+273.15).rho

    G1p1kocalc=Gcnf(3,1,ppgcn1,fpitgcn1,rohol1) #rash v m3/chas
    G1p1kmcalc=G1p1kocalc*rohol1/10**6 #rash v t/chas
    #G2p1kocalc=Gcnf(3,2,ppgcn1,fpitgcn1,rohol1) #rash v m3/chas
    #G2p1kmcalc=G2p1kocalc*rohol1/10**6 #rash v t/chas
    #G3p1kocalc=Gcnf(3,3,ppgcn1,fpitgcn1,rohol1) #rash v m3/chas
    #G3p1kmcalc=G3p1kocalc*rohol1/10**6 #rash v t/chas
    #G4p1kocalc=Gcnf(3,4,ppgcn1,fpitgcn1,rohol1) #rash v m3/chas
    #G4p1kmcalc=G4p1kocalc*rohol1/10**6 #rash v t/chas

    N1kcalcp1=G1p1kmcalc/3.6*(hgor1-hhol1)/10**3-nedgcn1/10**3
    #N1kcalcp2=G1p2kmcalc/3.6*(hgor2-hhol2)/10**3-nedgcn2/10**3
    #N1kcalcp3=G1p3kmcalc/3.6*(hgor3-hhol3)/10**3-nedgcn3/10**3
    #N1kcalcp4=G1p4kmcalc/3.6*(hgor4-hhol4)/10**3-nedgcn4/10**3
    #N1kcalc=N1kcalcp1+N1kcalcp2+N1kcalcp3+N1kcalcp4
    return N1kcalcp1

def der_tdgor(base=318.,std=1.):
    u"""
    погрешность мощности в зависимости от погрешности темп гор ниток
    """
    tgrand=np.random.normal(base,std,10000)
    #print tgrand.std()
    #print tgrand.mean()
    nd=[]
    for t in tgrand:
        nn=calcN1(tgorp1_sr=t)
        nd.append(nn)
    nd=np.array(nd)
    #print nd.mean()
    #print nd.std()
    #print nd.std()*100./nd.mean()
    return nd.std()*100./nd.mean()

def der_tdhol(base=287.,std=1.):
    u"""
    погрешность мощности в зависимости от погрешности темп хол ниток
    """
    tgrand=np.random.normal(base,std,10000)
    #print tgrand.std()
    #print tgrand.mean()
    nd=[]
    for t in tgrand:
        nn=calcN1(tholp1_sr=t)
        nd.append(nn)
    nd=np.array(nd)
    #print nd.mean()
    #print nd.std()
    #print nd.std()*100./nd.mean()
    return nd.std()*100./nd.mean()

def der_preak(base=15.6,std=0.11):
    u"""
    погрешность мощности в зависимости от погрешности давления в зоне
    """
    preakrand=np.random.normal(base,std,10000)
    #print tgrand.std()
    #print tgrand.mean()
    nd=[]
    for t in preakrand:
        nn=calcN1(preak=t)
        nd.append(nn)
    nd=np.array(nd)
    #print nd.mean()
    #print nd.std()
    #print nd.std()*100./nd.mean()
    return nd.std()*100./nd.mean()

def der_ppgcn1(base=0.58,std=0.005):
    u"""
    погрешность мощности в зависимости от погрешности давления в зоне
    """
    ppgcn1rand=np.random.normal(base,std,10000)
    #print tgrand.std()
    #print tgrand.mean()
    nd=[]
    for t in ppgcn1rand:
        nn=calcN1(ppgcn1=t)
        nd.append(nn)
    nd=np.array(nd)
    #print nd.mean()
    #print nd.std()
    #print nd.std()*100./nd.mean()
    return nd.std()*100./nd.mean()

nst=[]
for std in np.arange(0.1,5.,0.1):
    nst.append(der_tdgor(std=std))
nst=np.array(nst)
plt.plot(np.arange(0.1,5.,0.1),nst);plt.show()

nst=[]
for std in np.arange(0.1,5.,0.1):
    nst.append(der_tdhol(std=std))
nst=np.array(nst)
plt.plot(np.arange(0.1,5.,0.1),nst);plt.show()

nst=[]
for std in np.arange(0.05,1.5,0.05):
    nst.append(der_preak(std=std))
nst=np.array(nst)
plt.plot(np.arange(0.05,1.5,0.05),nst);plt.show()

nst=[]
for std in np.arange(0.005,0.5,0.005):
    nst.append(der_ppgcn1(std=std))
nst=np.array(nst)
plt.plot(np.arange(0.005,0.5,0.005),nst);plt.show()


def main():
    pass

if __name__ == '__main__':
    main()

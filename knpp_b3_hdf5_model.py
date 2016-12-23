#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from mc2py.giw import *
from mc2py.time_block2burnkamp import *
import datetime
import time
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rc
from freesteam import *
import matplotlib.ticker as ticker
import locale
from obrguess.model_drive import *
from obrguess.set2k import *
import numdifftools as nd

rc('font', **{'family': 'verdana'})
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

try:
    dir2='D:/work_place/serg/#b3-svbu/'
    store = pd.HDFStore(dir2+'store.h5')
except IOError:
    print u'Нет доступа к директории ',dir2

def out_param():
    u"""Возвращает словарь переменных модели"""
    aa=dict(Tgor1=v["YA11T810"],Tgor2=v["YA12T811"],Tgor3=v["YA13T812"],Tgor4=v["YA14T813"],
            Thol1=v["YA11T782"],Thol2=v["YA12T783"],Thol3=v["YA13T784"],Thol4=v["YA14T785"],
            dPgcn1=v["Y314B01"],dPgcn2=v["Y315B01"],dPgcn3=v["Y316B01"],dPgcn4=v["Y317B01"],
            Pzone1=v["Y065B022"],Pzone2=v["Y062B01"],
            Gp1kgs=v["yafloop1_G"],Gp2kgs=v["yafloop2_G"],Gp3kgs=v["yafloop3_G"],Gp4kgs=v["yafloop4_G"],
            Ntep=v["YMINTPOWMWT"],Naz=v["pow_az"],Nrr=v["pow_rr"],N1k=v["pow_1k"],N2k=v["pow_2k"],Naknp=v["ycintcammwt"],Nturb=v["N_TURB_RL"],
            Tpv1=v[u"14МЦ1-1"],Tpv2=v[u"14МЦ2-1"],Tpv3=v[u"14МЦ3-1"],Tpv4=v[u"14МЦ4-1"],
            Gpv1=v[u"14МЦ115Б"],Gpv2=v[u"14МЦ215Б"],Gpv3=v[u"14МЦ315Б"],Gpv4=v[u"14МЦ415Б"],
            Ppg1=v[u"14МЦ13"],Ppg2=v[u"14МЦ23"],Ppg3=v[u"14МЦ33"],Ppg4=v[u"14МЦ43"],
            Pgpk=v[u"19МЦ11"],tpvd1=v[u"M123"],tpvd2=v[u"M122"],ppvd1=v[u'P_pvda'],ppvd2=v[u'P_pvdb'],gpvd1=v[u'M603'],gpvd2=v[u'M604'],
            ppv1=v[u'Ppv1'],ppv2=v[u'Ppv2'],ppv3=v[u'Ppv3'],ppv4=v[u'Ppv4'],
            gkgtn=v[u'4RN81D01_G'])
    return aa

def out_model_coef():
    u"""возвращаем словарь коэффициентов модели"""
    coef_dict=dict(
    FLaSG1_CfResL=v['FLaSG1_CfResL'],
    FLaSG2_CfResL=v[u'FLaSG2_CfResL'],
    FLaSG3_CfResL=v[u'FLaSG3_CfResL'],
    FLaSG4_CfResL=v[u'FLaSG4_CfResL'],
    YD11D01_2_Hnom=v[u'YD11D01_2_Hnom'],
    YD12D01_2_Hnom=v[u'YD12D01_2_Hnom'],
    YD13D01_2_Hnom=v[u'YD13D01_2_Hnom'],
    YD14D01_2_Hnom=v[u'YD14D01_2_Hnom'],
    YHSIEVE_TUN=v[u'YHSIEVE_TUN(1)'],
    Loop_CfResL1=v[u'Loop_CfResL1'],
    Loop_CfResL2=v[u'Loop_CfResL2'],
    Loop_CfResL3=v[u'Loop_CfResL3'],
    Loop_CfResL4=v[u'Loop_CfResL4'],
    SG_CfResL1=v[u'SG_CfResL1'],
    SG_CfResL2=v[u'SG_CfResL2'],
    SG_CfResL3=v[u'SG_CfResL3'],
    SG_CfResL4=v[u'SG_CfResL4'],
    YhqCor1_eqf=v[u'YhqCor1_eqf'],
    YhqCor2_eqf=v[u'YhqCor2_eqf'],
    YhqCor3_eqf=v[u'YhqCor3_eqf'],
    Nin=v[u'YMINTPOW_SET'],
    Pgpkin=v[u'asut02oint1'],
    Tpvdain=v[u'PVDAT'],
    Tpvdbin=v[u'PVDBT'],
    Pazin=v[u'Ustavka_Pzone']
    )
    return coef_dict


def mean_fr_day(perem,pos):
    u"""Выбираем среднее значение за день по позиции"""
    poio=store['alldata'][perem].index[pos].isoformat()
    if(store['alldata'][perem][poio.split('T')[0]].mean()*0.05>store['alldata'][perem][poio.split('T')[0]].std()):
        valret=store['alldata'][perem][poio.split('T')[0]].mean()
    else: #возвращаем текущее если ошибка за день большая
        valret=np.NaN
    return valret

def put_data_fileh5_model(pos):
    u"""перекладываем данные из позиции базы в переменные модели
    y_data=get_y_data(0)
    """
    global store

    v.OG_N_pg=store['alldata']['Npg'][pos] #мощность по пг
    v.OG_N_pvd=store['alldata']['Npvd'][pos]
    v.OG_N_1k=store['alldata']['N1k'][pos]
    v.OG_N_gen=store['alldata']['Ngen'][pos]
    v.OG_N_aknp=store['alldata']['Naknp'][pos]
    v.OG_N_dpz=store['alldata']['Ndpz'][pos]
    v.OG_N_akz=store['alldata']['Nakz'][pos]
    v.OG_N_1k_calc=store['alldata']['N1kcalc'][pos]
    v.OG_N_pg_calc=store['alldata']['Npgcalc_prod'][pos]/10**6
    v.OG_N_pvd_calc=store['alldata']['Npvdcalc'][pos]/10**3

    v.OG_P_gpk=store['alldata']['pgpk'][pos]*10.197#Pgpk
    v.OG_T_gor=store['alldata'].iloc[pos][['tgorp1_sr','tgorp2_sr','tgorp3_sr','tgorp4_sr']]#tgor
    v.OG_T_hol=store['alldata'].iloc[pos][['tholp1_sr','tholp2_sr','tholp3_sr','tholp4_sr']]#thol
    v.OG_pp_gcn=store['alldata'].iloc[pos][['ppgcn1','ppgcn2','ppgcn3','ppgcn4']]*10.197#dpgcn
    v.OG_p_rea=store['alldata']['preak'][pos]*10.197#pzon
    v.OG_t_pitv=store['alldata'].iloc[pos][['tpitvpg1','tpitvpg2','tpitvpg3','tpitvpg4']]#tpv
    v.OG_g_pitv=store['alldata'].iloc[pos][['fpitv1','fpitv2','fpitv3','fpitv4']]#gpv
    v.OG_p_pitvg=store['alldata'].iloc[pos][['ppitv1','ppitv2','ppitv3','ppitv4']]#ppv
    v.OG_p_pg=store['alldata'].iloc[pos][['ppg1','ppg2','ppg3','ppg4']]*10.197#ppg
    v.OG_T_pvd1=store['alldata']['tpvd1_r'][pos]#T pvd1
    v.OG_T_pvd2=store['alldata']['tpvd2_r'][pos]#T pvd2
    v.OG_P_pvd=store['alldata'].iloc[pos][['ppvd1','ppvd2']]*10.197#ppvd
    v.OG_G_pvd=store['alldata'].iloc[pos][['fpvd1m','fpvd2m']]/1000.#gpvd
    v.OG_P_pvd_sr=store['alldata']['ppvd_r'][pos]*10.197#p pvd - na peremichke

    v.OG_arh_day=store["alldata"].index[pos].day
    v.OG_arh_month=store["alldata"].index[pos].month
    v.OG_arh_year=store["alldata"].index[pos].year
    v.OG_arh_hour=store["alldata"].index[pos].hour
    v.OG_arh_minute=store["alldata"].index[pos].minute
    v.OG_arh_second=store["alldata"].index[pos].second

def Teff_mas():
    u"""создаем массив нужного Тэфф"""
    global store
    teffmas=[]
    db_name='D:/work_place/serg/xipi.h5'
    for tt in store["alldata"].resample('D','mean').index:
        try:
            (kamp,teff)=get_kamp_burn(db_name=db_name, plant='kaln', block='b03', t=tt)
            teffmas.append([kamp,teff])
        except ValueError:
            print "ValueError, NO time in xipi.h5 ",tt
    return teffmas


solve_obr_guess=[] # массив решений задачи
shodsum_mas=[] #массив сходимости
shod_vec_mas=[]

def borfrN(N,Cbor):
    u"""Ищем бор от мощности"""
    print N,Cbor
    v.YMINTPOW_SET=N
    v.YM_STROBE_POWSET=1
    RET(5)
    while v.YM_STROBE_POWSET==1:RET()
    RET(5)
    print v.ymbor_cor-Cbor
    return v.ymbor_cor-Cbor

def set_bor(Cbor):
    u"""Устанавливаем состояние модели по концентрации бора"""
    start_model_drive(stat=False,boravt=False)
    ChooseNKSYA("NKS")
    SetCbRegulator(0)
    RET()
    o=scipy.optimize.brentq(borfrN,1.,120.,args=Cbor,rtol=0.001)
    SetCbRegulator(1)
    v.YMFLAGSTAT=0
    RET(3)
    ModelWait("abs(v.YMDRNEW)<0.000001")
    print u'реактивность1=',v.YMDRNEW
    ChooseNKSYA("YA")
    start_model_drive(stat=False,boravt=False)

def set_m_st(qqq):
    u"""Функция установки модели для линеаризации функции (поиска якобиана)
    требует массив из 5 переменных:
    N - мощность
    Pgpk - давление ГПК [атм]
    Tpvdа - температура в двух ПВД
    Tpvdb -
    Paz - давление в активной зоне"""
    global ijac
    global vecjac
    try:
        vecjac.append(qqq)
    except NameError:
        pass
    #(Cbor,Pgpk,Tpvda,Tpvdb)=qqq #otkaz ot bora dlya moshnosti
    #print Cbor,Pgpk,Tpvda,Tpvdb
    (N,Pgpk,Tpvda,Tpvdb,Paz)=qqq #otkaz ot bora dlya moshnosti
    print N,Pgpk,Tpvda,Tpvdb,Paz

    if v.ymintpow!=N:
        set_n(kamp=7,n=N,teff=0.5)
    start_model_drive(stat=False,boravt=False)
    RET(2)
##    print u"Ставим бор"
    #SetCbRegulator(state=0,dynamic=0,on=1)
    #v.YMFLAGSTAT=0
    #RET()
    #v.YMBOR_COR=Cbor
    #RET(10)
##    set_bor(Cbor)
    #ModelWait("abs(v.YMDRNEW)<0.000001")
    #print u'реактивность1=',v.YMDRNEW
    print u"Устанавливаем параметры второго контура"
    RET()
    ChooseNKSYA("YA")
    #v.YZKEYLINK2=1 #переход в полную модель
    RET()
    #while v.YZNKS_YA_TRAN:RET()
    start_model_drive(stat=True,boravt=True)
    start_model()
##    SetCbRegulator(1,dynamic=0,on=True)
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
##    if v.ymintpow!=N:RelaxState()
    if v.asut02oint1!=Pgpk:
        print u"Задаем уставку Р=",Pgpk,u"[атм] в ГПК...",
        v.asut02oint1=Pgpk #задаем уставку в ГПК
        print u"есть... Релаксируем...",
        RelaxByVar(u"19МЦ11")
        print u"есть"
    v.PVDT_flag=1 # Т ПВД берем напрямую
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    if v.PVDAT!=Tpvda:
        print u"Устанавливаем температуру=",Tpvda,u" в ПВД1","\t",
        v.PVDAT=Tpvda
        print u"есть... Релаксируем...",
        RelaxByVar(u"M123")
        print u"есть"
    if v.PVDBT!=Tpvdb:
        print u"Устанавливаем температуру=",Tpvdb,u" в ПВД2","\t",
        v.PVDBT=Tpvdb
        print u"есть... Релаксируем...",
        RelaxByVar(u"M122")
        print u"есть"
    if v.Ustavka_Pzone!=Paz or v.RegPodChast_ON==False:
        print u'Устанавливаем давление в АЗ=',Paz,"\t",
        v.Ustavka_Pzone=Paz
        v.RegPodChast_ON=True
        print u"есть... Релаксируем...",
        RelaxByVar(u"Y065B022")
        print u"есть"
    print u"Укачиваем...",
    RelaxDTDT(statmode=True)
    #RelaxState()
    v.reset_all=1
    print u"Закончили."
    SetCbRegulator(state=0,dynamic=0,on=1)
    '''{"Tgor":[v["YA11T810"],v["YA12T811"],v["YA13T812"],v["YA14T813"]],
            "Thol":[v["YA11T782"],v["YA12T783"],v["YA13T784"],v["YA14T785"]],
            "dPgcn":[v["Y314B01"],v["Y315B01"],v["Y316B01"],v["Y317B01"]],
            "Pzone":[v["Y065B022"],v["Y062B01"]],
            "Ntep":[v["YMINTPOWMWT"],v["pow_az"],v["pow_rr"],v["pow_1k"],v["pow_2k"],v["ycintcammwt"],v["N_TURB_RL"]],
            "Tpitv":[v[s2g(u"14МЦ1-1")],v[s2g(u"14МЦ2-1")],v[s2g(u"14МЦ3-1")],v[s2g(u"14МЦ4-1")]],
            "Gpitv":[v[s2g(u"14МЦ115Б")],v[s2g(u"14МЦ215Б")],v[s2g(u"14МЦ315Б")],v[s2g(u"14МЦ415Б")]],
            "Ppg":[v[s2g(u"14МЦ13")],v[s2g(u"14МЦ23")],v[s2g(u"14МЦ33")],v[s2g(u"14МЦ43")]]}'''
    try:
        ijac+=1
        print ijac,"\n"
    except NameError:
        pass
    return  out_param()


def set_m_st_deriv(qqq):
    u"""Функция установки модели для линеаризации функции (поиска якобиана)
    N - мощность в процентах
    Pgpk - давление ГПК [атм]
    Tpvd - температура в двух ПВД"""
    pass
    #dlya testa
    rf=open('D:/work_place/serg/obrguess/x02.npy')
    x0=np.loadtxt(rf)
    rf.close()
    #konec


def test_model_deriv():
    u"""проверяем как модель встает на состояние из разных условий
    то есть ищем погрешность модели"""
    global ijac,vecjac
    ijac=0
    vecjac=[]
    x0=np.array([100.,60.5,219.,219.])
    razm=10
    '''uniform distrib
    rcbor=np.random.rand(razm)*(8.3-8.)+8.
    rp=np.random.rand(razm)*(61.5-60.6)+60.6
    rtpvd1=np.random.rand(razm)*(225.-190.)+190.
    rtpvd2=np.random.rand(razm)*(225.-190.)+190.
    '''
    rn=np.random.normal(store['alldata']['Npg'][0:790].mean(),store['alldata']['Npg'][0:790].std(),razm)/31.2
    rp=np.random.normal(store['alldata']['pgpk'][0:790].mean()*10.197,store['alldata']['pgpk'][0:7000].std()*10.197,razm)
    rtpvd1=np.random.normal(210.,0.05,razm)
    rtpvd2=np.random.normal(208.,0.05,razm)
    rin=np.array([rn,rp,rtpvd1,rtpvd2]).T
    rout=[]
    x0out=[]
    for r in rin:
        rout.append(set_m_st(r))
        x0out.append(set_m_st(x0))

    x0outpr=np.array(x0out).T
    for q in x0outpr:
        print q.mean(),'\t',q.std(),'\t',q.std()*100/q.mean()

    af=open('D:/work_place/serg/obrguess/x02.npy','w')
    np.savetxt(af,x0out)
    af.close()
    af=open('D:/work_place/serg/obrguess/rout.npy','w')
    np.savetxt(af,rout)
    af.close()

def search_model_jac():
    ijac=0
    vecjac=[]
    Jfun_m = nd.Jacobian(set_m_st,step_max=0.05,step_ratio=1.1,step_num=12)
    x0=np.array([8.9,60.,219.,219.])
    jx0=Jfun_m(x0)

    af=open('D:/work_place/serg/obrguess/jac2.npy','w')
    np.savetxt(af,jx0)
    af.close()

    aff=open('D:/work_place/serg/obrguess/jac2.npy')
    jx0=np.loadtxt(aff)
    aff.close()

    p0=set_m_st(x0)
    xtest=np.array([9.,60.8,217.,215.])
    ftest=np.array([p0])

    distsize=100000
    xdelta=np.array([0.002,0.5,2.,2.])
    cbtest=np.random.normal(x0[0],xdelta[0],distsize)
    ptest=np.random.normal(x0[1],xdelta[1],distsize)
    tpvd1test=np.random.normal(x0[2],xdelta[2],distsize)
    tpvd2test=np.random.normal(x0[3],xdelta[3],distsize)
    xtest=np.column_stack([cbtest,ptest,tpvd1test,tpvd2test])
    ftest=[]
    for x in xtest:
        fnew=np.array([p0+np.dot(jx0,x-x0)])
        ftest.append(fnew)
    q=np.array(ftest)
'''    for i in range(10000):
        ntest=np.random.random_sample()*(9.-8.)+8.
        ptest=np.random.random_sample()*(61.5-55.5)+55.5
        tpvd1test=np.random.random_sample()*(220.-205.)+205.
        tpvd2test=np.random.random_sample()*(220.-205.)+205.
        xtest=np.array([ntest,ptest,tpvd1test,tpvd2test])
        fnew=np.array([p0+np.dot(jx0,xtest-x0)])
        ftest=np.append(ftest,fnew,axis=0)
'''

#    aarch=pd.DataFrame({"[100,61.,219.,219.]":a})
#    aguesh5=pd.HDFStore('D:/work_place/serg/obrguess/jac.h5')
#    aguesh5.put('solve',aarch)
#    aguesh5.close()


def shag_po_day_b3():
    u"""Шагаем по архиву усредненных значений и решаем обратную задачу
    """
    global store
    global shodsum_mas
    global solve_obr_guess
    db_name='D:/work_place/serg/xipi.h5'
    temp_ind=0
    start_model_drive()
    for tt in store["alldata"].resample('D','mean').index[2:3]:
        if store['alldata'].index.searchsorted(tt)!=temp_ind:
            try:
                print "\n"
                (kamp,teff)=get_kamp_burn(db_name=db_name, plant='kaln', block='b03', t=tt)
                kamp_i=int(kamp[1:])
                print tt
                ind_found=store['alldata'].index.searchsorted(tt)
                v["OG_Npoint(2)"]=int(ind_found)
                put_meddata(ind_found)
                RET()
                set_n(kamp=kamp_i,n=(v.OG_N_pg/30.),teff=teff)
                print u"Устанавливаем параметры второго контура"
                set2k()
                set_kgtn_g()
                print u"Решаем обр.задачу"
                sol,shod_sum,shod_vec=main_solve_obr_guess(v.OG_N_pg_calc/3000*100,v.OG_T_pvd1,v.OG_T_pvd2,v.OG_P_gpk)
                solve_obr_guess.append(sol)
                shodsum_mas.append(shod_sum)
                shod_vec_mas.append(shod_vec)
                pandaarch=pd.DataFrame({"solve_obr_guess":solve_obr_guess,"shodsum_mas":shodsum_mas,"shod_vec_mas":shod_vec_mas})
                obrguesh5=pd.HDFStore('D:/work_place/serg/obrguess/solve.h5')
                obrguesh5.put('solve',pandaarch)
                obrguesh5.close()
                #set_model_param_x(sol[0])
                temp_ind=store['alldata'].index.searchsorted(tt)
            except ValueError:
                print "ValueError, NO time in xipi.h5 ",tt


def put_meddata(pos):
    u"""перекладываем данные из позиции базы в переменные модели
    используем средние за день
    y_data=get_y_data(0)
    """
    global store

    v.OG_N_pg=mean_fr_day('Npg',pos) #мощность по пг
    v.OG_N_pvd=mean_fr_day('Npvd',pos)
    v.OG_N_1k=mean_fr_day('N1k',pos)
    v.OG_N_gen=store['alldata']['Ngen'][pos]
    v.OG_N_aknp=store['alldata']['Naknp'][pos]
    v.OG_N_dpz=store['alldata']['Ndpz'][pos]
    v.OG_N_akz=mean_fr_day('Nakz',pos)
    v.OG_N_1k_calc=store['alldata']['N1kcalc'][pos]
    v.OG_N_pg_calc=mean_fr_day('Npgcalc_prod',pos)/10**6
    v.OG_N_pvd_calc=store['alldata']['Npvdcalc'][pos]/10**3

    v.OG_P_gpk=store['alldata']['pgpk'][pos]*10.197#Pgpk
    v.OG_T_gor=store['alldata'].iloc[pos][['tgorp1_sr','tgorp2_sr','tgorp3_sr','tgorp4_sr']]#tgor
    v.OG_T_hol=store['alldata'].iloc[pos][['tholp1_sr','tholp2_sr','tholp3_sr','tholp4_sr']]#thol
    v.OG_pp_gcn=store['alldata'].iloc[pos][['ppgcn1','ppgcn2','ppgcn3','ppgcn4']]*10.197#dpgcn
    v.OG_p_rea=store['alldata']['preak'][pos]*10.197#pzon
    v.OG_t_pitv=store['alldata'].iloc[pos][['tpitvpg1','tpitvpg2','tpitvpg3','tpitvpg4']]#tpv
    v.OG_g_pitv=map(mean_fr_day,['fpitv1','fpitv2','fpitv3','fpitv4'],[pos,pos,pos,pos])#gpv
    v.OG_p_pitvg=store['alldata'].iloc[pos][['ppitv1','ppitv2','ppitv3','ppitv4']]*10.197#ppv
    v.OG_p_pg=store['alldata'].iloc[pos][['ppg1','ppg2','ppg3','ppg4']]*10.197#ppg
    v.OG_T_pvd1=store['alldata']['tpvd1_r'][pos]#T pvd1
    v.OG_T_pvd2=store['alldata']['tpvd2_r'][pos]#T pvd2
    v.OG_P_pvd=store['alldata'].iloc[pos][['ppvd1','ppvd2']]*10.197#ppvd
    v.OG_G_pvd=store['alldata'].iloc[pos][['fpvd1m','fpvd2m']]/1000.#gpvd
    v.OG_P_pvd_sr=store['alldata']['ppvd_r'][pos]*10.197#p pvd - na peremichke

    v.OG_arh_day=store["alldata"].index[pos].day
    v.OG_arh_month=store["alldata"].index[pos].month
    v.OG_arh_year=store["alldata"].index[pos].year
    v.OG_arh_hour=store["alldata"].index[pos].hour
    v.OG_arh_minute=store["alldata"].index[pos].minute
    v.OG_arh_second=store["alldata"].index[pos].second

    db_name='D:/work_place/serg/xipi.h5'
    (kamp,teff)=get_kamp_burn(db_name=db_name, plant='kaln', block='b03', t=store["alldata"].index[pos])
    kamp=int(kamp[1:])
    v.OG_arh_kamp=kamp
    v.OG_arh_teff=teff

##def put_graf():
##    global store
##    v.OG_Ngraf=store['alldata']['Npg'].values
##    aa=store['alldata']['Npg'].index.to_pydatetime()
##    bb=np.zeros(len(store['alldata']['Npg'].values))
##    for t in range(len(store['alldata']['Npg'].values)):
##        bb[t]=(time.mktime(aa[t].timetuple()) + aa[t].microsecond / 1E6)
##    bb=bb-bb[0]
##    v.OG_Ntgraf=bb
##    v.OG_Ntmin=min(bb)
##    v.OG_Ntmax=max(bb)
##    v.OG_Nmin=store['alldata']['Npg'].values.min()
##    v.OG_Nmax=store['alldata']['Npg'].values.max()
##    v.OG_Ncount=len(bb)
##    v.OG_Nflag+=1



if __name__ == '__main__':

    put_meddata(int(v.OG_Npoint[1]))
    #put_data_fileh5_model(int(v.OG_Npoint[1]))
    #shag_po_day_b3()
    ijac=0
    vecjac=[]
    xx=[8.9,60.0,219.0,219.0]
    xx1=[8.04095367706,61.163954738,200.875536102,196.992487997]
    set_m_st(xx)
    test_model_deriv()
    store.close()











def find_med_arch(perem):
    u"""Нормальное ли распределение perem"""
    rra=[] # границы кусочков на кот разбивают архивы
    rra.append(range(0,300))
    rra.append(range(301,500))
    rra.append(range(501,700))
    rra.append(range(854,1220))
    rra.append(range(1281,1649))
    rra.append(range(1769,3416))
    rra.append(range(3477,5122))
    rra.append(range(5183,5430))
    '''
    store['alldata']['Npg'][rra[0]].hist(bins=100)
    plt.xlabel(u'Мощность, МВт')
    plt.show()
    store['alldata']['Npg'][0:700].diff().hist(bins=100)
    plt.xlabel(u'Отклонение мощности, МВт')
    plt.show()
    '''
    store['alldata'][perem][rra[0]].hist()
    plt.show()
    bb=[]
    qq=[]
    for i in range(len(rra)):
        print i
        aa=store['alldata'][perem][rra[i]].diff()
        cc=np.array(aa[1:])
        dd=stats.mstats.normaltest(cc)
        bb.append(dd)
        qq.append(dd[1]>0.05) #обычный уровень значимости критерия пирсона
    return (bb,qq)














def oform_dis():
    u""" Псевдо программа для оформления диссертации
    """
    #11111111
    aa=store['alldata']['Npg']-store['alldata']['Npvd']
    bb=aa*100./store['alldata']['Npg']
    aa[:-2500].plot()
    plt.xlabel(u'Время')
    plt.ylabel(u'Разность мощностей, МВт')
    plt.show()
    bb[:-2500].plot()
    plt.xlabel(u'Время')
    plt.ylabel(u'Разность мощностей,%')
    plt.show()

    #222222222222222
    locale.setlocale(locale.LC_ALL,'rus')
    ddd=store['alldata'].iloc[:-2500][['tpitvpg1','tpitvpg2','tpitvpg3','tpitvpg4']]
    ddd=ddd.resample('D', how='mean')
    ddd.columns=[u'ПГ1',u'ПГ2',u'ПГ3',u'ПГ4']
    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(ddd.index,ddd.iloc[0:,0],'1',ddd.index,ddd.iloc[0:,1],'2',ddd.index,ddd.iloc[0:,2],'3',ddd.index,ddd.iloc[0:,3],'4')
    #ddd.plot(style='+')
    ax0.set_xlabel(u'Время')
    ax0.set_ylabel(u'Температура пит.воды, °С')
    ax0.legend([u'ПГ1',u'ПГ2',u'ПГ3',u'ПГ4'],loc=4)
    ax0.grid(1)

    ddpvd=store['alldata'][['tpvd1_r','tpvd2_r']][:-2500]
    ddpvd.name=u'Tпвд'
    ax1.plot(ddpvd.index,ddpvd.icol(0),'-',ddpvd.index,ddpvd.icol(1),'--')
    ax1.set_ylabel(u'Температура за ПВД, °С')
    def form_dat(x):
        return x.to_datetime()
#    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(form_dat))
    ax1.legend([u'ПВД А',u'ПВД Б'],loc=4)
    ax1.grid(1)
    fig.autofmt_xdate()
    plt.show()

    store['alldata'][['fpvd1m','fpvd2m']].plot()
    plt.show()

    #2222233333
    fpitv=store['alldata'][['fpitv1','fpitv2','fpitv3','fpitv4']].sum(1)#F ПВ ПГ1 YB10W01 0/2000 Т/Ч
    fpvd=store['alldata'][['fpvd1','fpvd2']].sum(1)#F ПВД ГРУП А 0/5600 М3/Ч
    fpvdm=store['alldata'][['fpvd1m','fpvd2m']].sum(1)#F ПВД ГРУП А КГ/Ч
    fpvdm=fpvdm/1000 #Т/Ч
    fkgtn=fpitv-fpvdm
    fkgtn_cut=fkgtn[fkgtn<1500]
    fkgtn_cut=fkgtn_cut[fkgtn_cut>400]
    fkgtn_cut.plot()
    plt.xlabel(u'Время')
    plt.ylabel(u'Рассчитанный расход от КГТН, т/ч')
    plt.show()
    fkgtn_cut.describe()
    fpvdm[0]/3.6 #кг/с
    store['alldata']['Npvd'][0]/(fpvdm[0]/3.6)
    (store['alldata']['Npvd'][0]/(fpvdm[0]/3.6))*((fpvdm[0]/3.6)+fkgtn_cut.mean()/3.6)-store['alldata']['Npvd'][0]
    Npvd_kgtn_dif=(store['alldata']['Npvd']/(fpvdm/3.6))*((fpvdm/3.6)+fkgtn_cut.mean()/3.6)-store['alldata']['Npvd']
    Npvd_kgtn=(store['alldata']['Npvd']/(fpvdm/3.6))*((fpvdm/3.6)+fkgtn_cut.mean()/3.6)
    Npvd_kgtn.plot()
    store['alldata']['N1k'].plot()
    store['alldata']['Npg'].plot()
    store['alldata']['Npvd'].plot()
    plt.show()

    fkgtn_cut.mean()/3.6 #кг/с

    fkgtn_mean=fkgtn.resample('5D', how='mean')



    #333333333333
    store['alldata'].iloc[:-2500][['fpitv1','fpitv2','fpitv3','fpitv4']].plot()
    plt.show()
    store['alldata'].iloc[:-2500][['ppitv1','ppitv2','ppitv3','ppitv4']].plot()
    plt.show()


    #4444444444444
    h1=steam_pT(store['alldata']['ppitv1'][1]*10**6,store['alldata']['tpitvpg1'][1]+273.15).h
    Tsat1=Tsat_p(store['alldata']['ppg1'][1]*10**6)
    h2=steam_Tx(Tsat1,1).h
    g1=store['alldata']['fpitv1'][1]/3.6
    n1=g1*(h2-h1)/10**6

    h11=steam_pT(store['alldata']['ppitv1'][1]*10**6,store['alldata']['tpitvpg1'][1]+273.15+5).h
    Tsat11=Tsat_p(store['alldata']['ppg1'][1]*10**6)
    h21=steam_Tx(Tsat1,1).h
    g11=store['alldata']['fpitv1'][1]/3.6
    n11=g11*(h21-h11)/10**6

    difN2k=store['alldata']['Npg'][-3000:]-store['alldata']['Npvd'][-3000:]
    plt.plot(store['alldata']['Npg'][-3000:],difN2k,'r+')
    plt.xlabel(u'Nпг, МВт')
    plt.ylabel(u'Nпг-Npvd, МВт')
    plt.grid(1)
    plt.show()

    #sistem pogresh ot KGTN
    sispog_kgtn=(store['alldata']['Npg']-store['alldata']['Npvd'])/store['alldata']['Npg']*100
    sispog_kgtn.describe()

    #perepad T na petlyah
    perep=[v.YA11T810-v.YA11T782,
    v.YA12T811-v.YA12T783,
    v.YA13T812-v.YA13T784,
    v.YA14T813-v.YA14T785]

    #555555555
    # Рассогласование датчиков первой петли горяч температур

    tgorp1name=['tgorp1_sr','tgorp1_s5dtc','tgorp1_s4da','tgorp1_s4db','tgorp1_s4dc','tgorp1_s3da','tgorp1_s3db','tgorp1_s3dc','tgorp1_s2da','tgorp1_s2db','tgorp1_s1da','tgorp1_s1db','tgorp1_s1dc']
    store['alldata'][tgorp1name].plot()
    plt.show()



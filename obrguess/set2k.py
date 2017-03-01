#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from mc2py.giw import *
import datetime
import time
import numpy as np
from mc2py.showkart import kshow
from obrguess.model_drive import *

def set2k():
    u"""Установка параметров второго контура модели по данным
    """
    v.YZKEYLINK2=1 #переход в полную модель
    RET()
    while v.YZNKS_YA_TRAN:RET()
    start_model_drive()
    start_model()
    RelaxState()
    SetCbRegulator(1,dynamic=0,on=True)
    print u"Устанавливаем мощность ",v.OG_N_pg/3000*100,u"[%] "
    v.YMINTPOW_SET=v.OG_N_pg/3000*100
    #v.YMFLAGSTAT=0 #выключаем статику
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    print u"Задаем уставку Р=",v.OG_P_gpk,u"[атм] в ГПК"
    v.asut02oint1=v.OG_P_gpk #задаем уставку в ГПК
    RelaxByVar(s2g(u"19МЦ11"))
    v.PVDT_flag=1
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    print u"Устанавливаем температуру=",v.OG_T_pvd1,u" в ПВД1"
    v.PVDAT=v.OG_T_pvd1
    v.PVDBT=v.OG_T_pvd2
#    set_perem(0.99,"cf_rlnvda",t[0],s2g(u"M123"))
    RelaxByVar(s2g(u"M123"))
    #pr_dtdt()
    print u"Устанавливаем температуру=",v.OG_T_pvd2,u" в ПВД2"
#    set_perem(0.99,"cf_rlnvdb",t[1],s2g(u"M122"))
    RelaxByVar(s2g(u"M122"))
    #pr_dtdt()
    v.reset_all=1
    print u"Закончили"
    SetCbRegulator(state=0,dynamic=0,on=1)
    return

def set_kgtn_g():
    u"""Устанавливаем расход КГТН
    """
    #v.res_PvdaOut_Res0=0.1
    #v.res_PvdbOut_Res0=0.1

    g_kgtn=v.OG_g_pitv.sum()-v.OG_G_pvd.sum() #m3/ch
    g_kgtn=g_kgtn*785/3600 #kg/s
    v.OG_G_kgtn=g_kgtn

    #sol2=optimize.fsolve(find_perem,0.1,args=('4RN81D01',138.0,'4RN81D01_G',v),factor=10)
    #sol2=optimize.minimize_scalar(find_perem,args=('4RN81D01',138.0,'4RN81D01_G',v),bounds=(0.1,1),method='bounded')
    sol3=optimize.brentq(find_perem,0.1,1.0,args=('4RN81D01',g_kgtn,'4RN81D01_G',v),xtol=0.01)
    v['4RN81D01']=sol3

def vliyanie_kgtn():
    set2k()
    #1 KGTN peremeshivaetsya ravnomerno
    v.res_kgtn_rl31_Res0=9999
    v.res_kgtn_rl32_Res0=9999
    v.res_kgtn_rl33_Res0=9999
    v.res_kgtn_rl34_Res0=9999
    v.res_kgtn_pvda_Res0=1
    v.res_kgtn_pvdb_Res0=1
    RelaxByVar(s2g(u"14МЦ4-1"))
    n1k_ravn=v.pow_1k
    npg_ravn=v.pow_2k
    nturb_ravn=v.N_TURB_RL
    gpv_ravn=np.ones(4)
    gpv_ravn[0]=v[s2g(u"14МЦ115Б")]
    gpv_ravn[1]=v[s2g(u"14МЦ215Б")]
    gpv_ravn[2]=v[s2g(u"14МЦ315Б")]
    gpv_ravn[3]=v[s2g(u"14МЦ415Б")]
    gprod_ravn=np.ones(4)
    gprod_ravn[0]=v.RY11F01
    gprod_ravn[1]=v.RY11F02
    gprod_ravn[2]=v.RY11F03
    gprod_ravn[3]=v.RY11F04
    Ppg4_ravn=v.YB14W01_Pw
    Ppg1_ravn=v.YB11W01_Pw
    Ppg3_ravn=v.YB13W01_Pw
    Ppg2_ravn=v.YB12W01_Pw
    Entpg4_ravn=v.YB14W01_EeWt
    gkgtn_ravn=v['4RN81D01_G']
    Kq_ravn=v.PAR_VALUE
    ql_ravn=v.PAR_VALUE
    qtvs_ravn=v.ysvrk_nka
    ql7_ravn=v.YSVRK_QL7

    #2 KGTN peremeshivaetsya v PG4
    v.res_kgtn_rl31_Res0=9999
    v.res_kgtn_rl32_Res0=9999
    v.res_kgtn_rl33_Res0=9999
    v.res_kgtn_rl34_Res0=40
    v.res_kgtn_pvda_Res0=99
    v.res_kgtn_pvdb_Res0=99
    RelaxByVar(s2g(u"14МЦ4-1"))
    n1k_4pg=v.pow_1k
    npg_4pg=v.pow_2k
    nturb_4pg=v.N_TURB_RL
    gpv_4pg=np.ones(4)
    gpv_4pg[0]=v[s2g(u"14МЦ115Б")]
    gpv_4pg[1]=v[s2g(u"14МЦ215Б")]
    gpv_4pg[2]=v[s2g(u"14МЦ315Б")]
    gpv_4pg[3]=v[s2g(u"14МЦ415Б")]
    gprod_4pg=np.ones(4)
    gprod_4pg[0]=v.RY11F01
    gprod_4pg[1]=v.RY11F02
    gprod_4pg[2]=v.RY11F03
    gprod_4pg[3]=v.RY11F04
    Ppg4_4pg=v.YB14W01_Pw
    Ppg1_4pg=v.YB11W01_Pw
    Ppg3_4pg=v.YB13W01_Pw
    Ppg2_4pg=v.YB12W01_Pw
    Entpg4_4pg=v.YB14W01_EeWt
    gkgtn_4pg=v['4RN81D01_G']
    Kq_4pg=v.PAR_VALUE
    ql_4pg=v.PAR_VALUE
    qtvs_4pg=v.ysvrk_nka
    ql7_4pg=v.YSVRK_QL7
    #kshow((Kq_4pg-Kq_ravn)/Kq_4pg*100,show=1,updn=1,format='%1.2g',gray=True)
    kshow((ql_4pg-ql_ravn),show=1,updn=1,format='%1.2g')
    kshow((qtvs_4pg-qtvs_ravn)/qtvs_ravn*100,show=1,updn=1,format='%1.2g',gray=True,dnLim=0.1, upLim=.3)
    print ql7_4pg.max()-ql7_ravn.max()
    print (ql7_4pg.max()-ql7_ravn.max())/ql7_4pg.max()*100

if __name__ == '__main__':
    set2k()

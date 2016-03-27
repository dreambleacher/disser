#!/usr/bin/env python
# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name: Линеаризация модели станции
# Purpose: Найти коэффициенты изменения датчиков модели в зависимости от изменения параметров модели
#
# Author:      iliyas
#
# Created:     08.03.2016
# Copyright:   (c) iliyas 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from knpp_b3_hdf5_model import *

u"""
разделим переменные, которые используем в модели
первая группа уже исследованных
из def_vars mod_coef
вторая группа начальных условий
N
Tpvd1
Tpvd2
Pgpk
Paz
Gkgtn
res KGT (мильен сопротивлений кгтна)
out_model_coef()
"""

dirofcalc='G:/git_disser/disser/'

def create_net_coef():
    u"""
    создаем сеточку параметров модели для расчета якобиана
    """
    q_mc=5 #число разбиения отрезка параметров
    mod_coef_net={}
    for m_c in mod_coef:
        mod_delta = mod_coef_delta[m_c][1] - mod_coef_delta[m_c][0]
        mod_coef_net[m_c] = np.linspace(mod_coef_delta[m_c][0],mod_coef_delta[m_c][1],q_mc)
    return mod_coef_net

def create_net_vhod():
    u"""
    создаем сеточку входов модели для расчета якобиана
    """
    q_mc=5 #число разбиения отрезка параметров
    mod_vhod_net={}
    for v_c in vh_param:
        mod_delta = vh_delt[v_c][1] - vh_delt[v_c][0]
        mod_vhod_net[v_c] = np.linspace(vh_delt[v_c][0],vh_delt[v_c][1],q_mc)
    return mod_vhod_net

def coef_input(inmas):
    u"""
    передем в модель значения параметров модели
    """
    global mod_coef
    if len(inmas)!=len(mod_coef):
        error_msg = 'Wrong input massive'
        raise ValueError, error_msg
    for i in range(len(mod_coef)):
        v[mod_coef[i]]=inmas[i]


def process_linearise():
    u"""
    Тело где выставляем модель по параметрам
    """
    param_vh = np.array([100.,60.9,215.,215.,160.]) #temp! только до того как запишем это параметром в функцию
    param_mas = []#temp! только до того как запишем это параметром в функцию
    for mc in mod_coef:#temp! только до того как запишем это параметром в функцию
        param_mas.append(x0[mc])#temp! только до того как запишем это параметром в функцию
    #v.OPCCMD=4 #load(!) state#загрузка состояния
    coef_input(param_mas)#запись коэффициентов в модель
    set_m_st(param_vh)#расчет модели по 5 параметрам
    #oudict=out_param()#вывод датчиков
    #coef_dict=out_model_coef()#вывод параметров модели
    #запись в файл

def first_run(filename='liner_JAC_model_x0_by1.h5'):
    'create h5 massive'


    v.OPCCMD=4 #load(!) state
    oudict=out_param()
    outinp=out_model_coef()
    storeofd = pd.HDFStore(dirofcalc+filename)
    storeofd['model_data']=pd.DataFrame(oudict,index=[0])
    storeofd['inp_data']=pd.DataFrame(outinp,index=[0])
    storeofd.close()


def go_throught_net(param_vhm=[100.,60.9,215.,215.,160.],filename='liner_JAC_model_x0_by1.h5'):
    u"""
    идем по сетке
    """
    startall = time.time()
    param_vh = np.array(param_vhm)
    for m_c in mod_coef: #для каждого параметра модели
        for pp in mod_coef_net[m_c]:
            start = time.time()
            print '____________________________________________________'
            print m_c,pp
            v.OPCCMD=4 # load(!) state для отката
            v[m_c]=pp  # меняется 1 параметр на каждом шаге, что есть возмущение модели
            set_m_st(param_vh)
            oudict=out_param()
            coef_dict=out_model_coef()
            storeofd = pd.HDFStore(dirofcalc+filename)
            storeofd['model_data']=storeofd['model_data'].append(oudict,ignore_index=True)
            storeofd['inp_data']=storeofd['inp_data'].append(coef_dict,ignore_index=True)
            storeofd.close()
            finish = time.time()
            print u"Скорость выполенения: ",(finish - start)/60.,u" минут"
    for vh_c in vh_param: #для каждого входного данного модели
        for vv in mod_vhod_net[vh_c]:
            start = time.time()
            print '____________________________________________________'
            print vh_c,vv
            vh0=dict(N=param_vhm[0],Pgpk=param_vhm[1],tpvd1=param_vhm[2],tpvd2=param_vhm[3],Paz=param_vhm[4])
            vh0[vh_c]=vv
            vhmas=np.array([vh0['N'],vh0['Pgpk'],vh0['tpvd1'],vh0['tpvd2'],vh0['Paz']])
            print vhmas
            v.OPCCMD=4 # load(!) state для отката
            # добавить расчет модели с изменением только 1 переменной!
            set_m_st(vhmas)
            oudict=out_param()
            coef_dict=out_model_coef()
            storeofd = pd.HDFStore(dirofcalc+filename)
            storeofd['model_data']=storeofd['model_data'].append(oudict,ignore_index=True)
            storeofd['inp_data']=storeofd['inp_data'].append(coef_dict,ignore_index=True)
            storeofd.close()
            finish = time.time()
            print u"Скорость выполенения: ",(finish - start)/60.,u" минут"
    finishall = time.time()
    print u"Скорость выполенения всего: ",(finishall - startall)/60.,u" минут"

def net4net():
    u"""
    необходимо проверить зависимость якобиана
    от начальной точки в которой считаем якобиан модели
    """
    mas4test=[  [98.,60.9,215.,215.,160.],
                [100.,61.3,215,215.,160.],
                [100.,60.9,220,215.,160.],
                [100.,60.9,215.,215.,159.]]
    for mt in mas4test:
        print mt
        fn='liner_JAC_model_x0'+str(param_vhm)[1:-1]+'.h5'
        first_run(fn)
        go_throught_net(param_vhm=mt,filename=fn)



def main():
    pass

if __name__ == '__main__':
    main()

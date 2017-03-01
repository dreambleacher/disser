#!/usr/bin/env python
# -*- coding: cp1251 -*-

import sys
##try:
##    #from mc2py.mopc import *
##except:
##    try:
##        print "1",sys.exc_info()
##
##    print "1"
##    time.sleep(5)
from mc2py.giw import *
import time
from scipy import optimize
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from py.knpp_b3_hdf5_model import *


def start_model():
    ModelStart()
    '''timstart=v["$_MDLTIME"]
    #time.sleep(1)
    while v["$_MODELISSTOPPED"]:
        v.OPCCMD=12 #start model
     #   time.sleep(0.5)'''

def start_model_drive(stat=True,boravt=True):
    u"""программа для перевода модели в режим для решения обр задачи"""

    if stat:v.YMFLAGSTAT=1 #переход в статику
    v.ym_flag_ake=1 #идеальный АКЭ
    v.YZBORREG=1   #включаем борный регулятор
    v.YmwNom=3000000000.0
    time.sleep(0.5) #иначе регулятор не переводится в автомат почему-то :(
    if boravt:v.YZBORMODE=2  #ставим его в автомат
    v.ARM_KEY_POS=2 #выключаем АРМ
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    v.HB113T1_KEY=1 #блокируем АЗ1
    v.HB113T55_KEY=1 #блокируем АЗ3
    v.HB113T77_KEY=1 #блокируем АЗ4
    v.HB113T65_KEY=1 #блокируем РОМ
    v.HB113T68_KEY=1 #блокируем УРБ
    v.KEY_BLKBLK=1 #блокируем блокировки :)
    v.noise_PGREG=0 #выключаем шум в регуляторах ПГ
    #v.rlregpg2c1=0 #убираем шум в датчиках парогенераторов
    #v.rlregpg2c2=0 #убираем шум в датчиках парогенераторов
    #v.rlregpg3c1=0 #убираем шум в датчиках парогенераторов
    #v.rlregpg3c2=0 #убираем шум в датчиках парогенераторов
    v.reset_all=1 #сброс всей сигнализации

def vigoraem():
    u"""Выгораем моделью и делаем состояния по нужным TEFF"""
    v.YZKEYLINK2=0
    v.YMFLAGSTAT=1
    print u"Создаем файлы по которым потом горим и делаем состояния"
    tefmas=np.array(Teff_mas())
    ww=pd.DataFrame(tefmas)
    ww[1]=ww[1].astype('float64').round(2)

    for ind in ww[0].drop_duplicates():
        ee=ww[ww[0]==ind][1].tolist()
        xipidir="D:\work_place\serg\mfa3_f13\kln_rea_vig\MFA-RD-KNPP-1000\VVER1000\INPUT\XIPI1\kaln\B03"
        fxipi=open(xipidir+"/"+"k0"+str(int(ind[1:])+1)+"/nut_prn_tim",'w')
        val1=ee
        str1="kartog_prn_tim="
        for vv in val1:
            str1+=str(vv)
            str1+=', '
        str2="state_save_tim="
        for vv in val1:
            str2+=str(vv)
            str2+=', '
        fxipi.write(str1[:-2]+'\n')
        fxipi.write(str2[:-2])
        fxipi.close()


def set_teff(teff):
    u"""догараем до нужных суток и ставим на паузу"""
    print u"Ставим Тэфф = ",teff
    if v.YM_FLAG_GRN_SIM_PAUSE==1 and v.YMTIME_BRN-teff<=5.: #Если можем дошагать с паузы до след выгорания
        v.YM_FLAG_GRN_SIM_PAUSE=0
        while(v.YMTIME_BRN<=teff):
            RET()
        v.YM_FLAG_GRN_SIM_PAUSE=1
        v.YMFAST=1
    else: #если нет, то начинаем заново гореть
        print u"Горим сначала"
        v.YM_XIPI_GLBSIM=0
        RET(5)
        v.ym_autostpsize_flag=1
        v.YM_XIPI_LDBRNBEG=1
        RET(5)
        v.YM_XIPI_GLBSIM=1
        RET(5)
        while(v.YMTIME_BRN<=teff):
            RET()
        v.YM_FLAG_GRN_SIM_PAUSE=1
        v.YMFAST=1
    print u"Установили = ",v.YMTIME_BRN

def set_n(kamp,n,teff,blk=3):
    u"""Устанавливаем параметры кампании, блока и мощность с Тэфф"""
    if blk==3: kamp=kamp+1
    if v.YMBLOCKNUMBER0!=blk or kamp!=v.YMLOADNUMBER:
        v.YZKEYLINK2=0
        v.YMFLAGSTAT=1
        v.YM_XIPI_GLBSIM=0
        print u"Ставим блок = ",blk
        v.YMBLOCKNUMBER_TO_LOAD=blk
        print u"Ставим кампанию = ",kamp
        v.YM_N_KAMP_TO_LOAD=kamp
        v.YM_XIPI_LDBRNBEG=1
        RET(25)
    v.YZKEYLINK2=0
    v.YMFLAGSTAT=1
    set_teff(teff)
    print u"Ставим мощность = ",n
    v.YMINTPOW_SET=n
    v.YM_STROBE_POWSET=1
    RET(25)


def pr_dtdt():
    u"""программа укачки модели"""

    RelaxDTDT(size=50.)
    u'''оставили для обратной совместимости
    print u"Начало программы укачки"
    print 'dt1kdt=',v.dt1kdt
    meastime=v["$_MDLTIME"]
    time.sleep(1)
    dtdt=abs(v.dt1kdt)
    while (dtdt>0.2)or(abs(v.dt1kdt)>0.1):
        #print v.dt1kdt
        for i in range(1,49):
            mdltime_t=v["$_MDLTIME"]
            while v["$_MDLTIME"]-mdltime_t==0:
                time.sleep(0.0001)
            yy=i/(i+1.)
            dtdt=dtdt*yy + abs(v.dt1kdt)*(1-yy)
            #print 'yy=',yy,'   dtdt=',dtdt,'   dt1kdt=',v.dt1kdt
    print u"укачка закончена. T=",v["$_MDLTIME"]-meastime," c"
    '''
def gpetl_set(x,y,v=v):
    u"""
    Установка расхода в петлях по данным
    x - 4 коэфициента регулировочных сопротивлений в петле
    y - 4 нужных расхода
    функция возвращает 4 разности между расходами при x и расходами y
    """
    v.OPCCMD=8 #save(!) state
    xbak=x
    v.Loop_CfResL1=x[0]
    v.Loop_CfResL2=x[1]
    v.Loop_CfResL3=x[2]
    v.Loop_CfResL4=x[3]
    pr_dtdt() #в теории нужно укачивать модель, но пока ждем 2 сек
    #time.sleep(4)
    retvar=[v.yafloop1_G-y[0],v.yafloop2_G-y[1],v.yafloop3_G-y[2],v.yafloop4_G-y[3]]
    v.OPCCMD=4 #load(!) state
    x=xbak
    start_model()
    return retvar

def find_perem(param_value,param,perem_value,perem,v=v):
    u"""Поиск переменной модели по параметру возвращает разницу между нужным значением переменной и измененной
    perem - имя переменной модели (датчика) которую хотим получить
    perem_value - значение переменной к которой в итоге приводим модель
    param - имя управляющего параметра модели
    param_value - новое значение управляющего параметра
    """
    print param_value,param,perem_value,perem
    v.OPCCMD=8 #save(!) state
    param_bak=v[param]
    v[param]=param_value #0 потому что передает из set-perem в виде массива
    RelaxByVar(param)
    #pr_dtdt() #в теории нужно укачивать модель, но пока ждем 2 сек
    #time.sleep(10)
    retv=v[perem]-perem_value
    v.OPCCMD=4 #load(!) state
    v[param]=param_bak
    start_model()
    print retv
    return retv

def set_perem(param_value,param,perem_value,perem,v=v,xtol=0.01):
    u"""Высталение переменной модели по параметру возвращает разницу между нужным значением переменной и измененной
    perem - имя переменной модели (датчика) которую хотим получить
    perem_value - значение переменной к которой в итоге приводим модель
    param - имя управляющего параметра модели
    param_value - новое значение управляющего параметра
    """
    sol2=optimize.fsolve(find_perem,param_value,args=(param,perem_value,perem,v),xtol=xtol)
    v[param]=sol2[0]

def s2g(str):
    u"""перекодировка строки из cp1251 в latin1"""
    ##return str.encode("cp1251").decode("latin1")
    return str #giw rabotaet uge bez perecod

def set_ntp(n,t,p,blk=3,kmp=7,v=v):
    u"""Установка модели по данным
    n - мощность в [%]
    t - температура ПВ[0,1] [град С]
    p - давление в ГПК [атм]
    blk - номер блока
    kmp - номер кампании
    """
    SetCbRegulator(1,dynamic=0,on=True)
    print u"Устанавливаем мощность ",n,u"[%] "
    v.YMINTPOW_SET=n
    #SetMFADrive(n,block=blk,kamp=kmp) #установка мощности для 3 блока и 7 кампании
    v.YMFLAGSTAT=0 #выключаем статику
    print u"Задаем уставку Р=",p,u"[атм] в ГПК"
    v.asut02oint1=p #задаем уставку в ГПК
    RelaxByVar(s2g(u"19МЦ11"))
    #pr_dtdt()
    print u"Устанавливаем температуру=",t[0],u" в ПВД1"
    set_perem(0.99,"cf_rlnvda",t[0],s2g(u"M123"))
    RelaxByVar(s2g(u"M123"))
    #pr_dtdt()
    print u"Устанавливаем температуру=",t[1],u" в ПВД2"
    set_perem(0.99,"cf_rlnvdb",t[1],s2g(u"M122"))
    RelaxByVar(s2g(u"M122"))
    #pr_dtdt()
    v.reset_all=1
    print u"Закончили"
    SetCbRegulator(state=0,dynamic=0,on=1)
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    return {"Tgor":[v["YA11T810"],v["YA12T811"],v["YA13T812"],v["YA14T813"]],
            "Thol":[v["YA11T782"],v["YA12T783"],v["YA13T784"],v["YA14T785"]],
            "dPgcn":[v["Y314B01"],v["Y315B01"],v["Y316B01"],v["Y317B01"]],
            "Pzone":[v["Y065B022"],v["Y062B01"]],
            "Ntep":v["YMINTPOWMWT"],
            "Tpitv":[v[s2g(u"14МЦ1-1")],v[s2g(u"14МЦ2-1")],v[s2g(u"14МЦ3-1")],v[s2g(u"14МЦ4-1")]],
            "Gpitv":[v[s2g(u"14МЦ115Б")],v[s2g(u"14МЦ215Б")],v[s2g(u"14МЦ315Б")],v[s2g(u"14МЦ415Б")]],
            "Ppg":[v[s2g(u"14МЦ13")],v[s2g(u"14МЦ23")],v[s2g(u"14МЦ33")],v[s2g(u"14МЦ43")]]}


def show_inmfa(pos):
    u"""Показываем на формате SPR в секции 4 данные из pos
    pos - позиция архива
    """
    v["YhS_Tcool"]=[tholp1_sr[pos],tholp2_sr[pos],tholp3_sr[pos],tholp4_sr[pos]]
    v["YhS_Thot"]=[tgorp1_sr[pos],tgorp2_sr[pos],tgorp3_sr[pos],tgorp4_sr[pos]]
    v["YhS_dPgcn"]=ppgcn.ix[pos]*10.197
    v["YhS_Ppg_spr"]=ppg.ix[pos]*10.197
    v["Yhs_gfw"]=fpitv.ix[pos]
    v["Yhs_TAvrSg"]=(tholp1_sr[pos]+tholp2_sr[pos]+tholp3_sr[pos]+tholp4_sr[pos])/4
    v["Yhs_GSumSg"]=fpitv.ix[pos].sum()

def get_y_data(pos):
    u"""На какие показания датчиков будем натягивать модель
    y_data=get_y_data(0)
    """
    y2y =[]
    y2y.append([tgorp1_sr[pos],tgorp2_sr[pos],tgorp3_sr[pos],tgorp4_sr[pos]])#tgor
    y2y.append([tholp1_sr[pos],tholp2_sr[pos],tholp3_sr[pos],tholp4_sr[pos]])#thol
    y2y.append((ppgcn.ix[pos].values*10.197).tolist())#dpgcn
    y2y.append([preak.ix[pos]*10.197])#pzon
    y2y.append(tpitvpg.ix[pos].values.tolist())#tpv
    y2y.append(fpitv.ix[pos].values.tolist())#gpv
    y2y.append((ppg.ix[pos].values*10.197).tolist())#ppg
    return sum(y2y,[])

def get_y_model():
    u"""берем из формата в модели показания датчиков на которые натягиваем модель
    """
    yfrmod=np.array([])
    yfrmod=np.append(yfrmod,v.OG_T_gor)#tgor
    yfrmod=np.append(yfrmod,v.OG_T_hol)#thol
    yfrmod=np.append(yfrmod,v.OG_pp_gcn)#dpgcn
    yfrmod=np.append(yfrmod,v.OG_p_rea)#pzon
    yfrmod=np.append(yfrmod,v.OG_t_pitv)#tpv
    yfrmod=np.append(yfrmod,v.OG_g_pitv)#gpv
    yfrmod=np.append(yfrmod,v.OG_p_pg)#ppg
    return yfrmod

u"""массив исходных значений коэффициентов модели"""
x0_obr_func=[1.6,1.6,1.6,1.6,
            86.,86.,86.,86.,
            1.1,
            0.1,0.1,0.1,0.1,
            2.5,2.5,2.5,2.5,
            0.1,0.1,0.1,]


u"""24 параметр -1 = 23 убрали бор - 3 =20 убрали гпк и пвд"""
x_model=[#'sgoutres1', #коэф сопр на выходе из ПГ
#'sgoutres2', #коэф сопр на выходе из ПГ
#'sgoutres3', #коэф сопр на выходе из ПГ
#'sgoutres4', #коэф сопр на выходе из ПГ
'FLaSG1_CfResL',#коэф сопр на выходе из ПГ за датчиком P
'FLaSG2_CfResL',
'FLaSG3_CfResL',
'FLaSG4_CfResL',
'YD11D01_Hnom', #Коэфф.настройки гомологической характеристики ГЦН-1
'YD12D01_Hnom', #Коэфф.настройки гомологической характеристики ГЦН-2
'YD13D01_Hnom', #Коэфф.настройки гомологической характеристики ГЦН-3
'YD14D01_Hnom', #Коэфф.настройки гомологической характеристики ГЦН-4
'YHSIEVE_TUN', #Коэф рег сопр АЗ - массив нужен 0 элемент!
'Loop_CfResL1',#+val #Коэф. регул сопр петли
'Loop_CfResL2',#+val #Коэф. регул сопр петли
'Loop_CfResL3',#+val #Коэф. регул сопр петли
'Loop_CfResL4',#+val #Коэф. регул сопр петли
'SG_CfResL1',#Коэфф.рег.сопр.тр.ПГ1
'SG_CfResL2', #Коэфф.рег.сопр.тр.ПГ2
'SG_CfResL3', #Коэфф.рег.сопр.тр.ПГ3
'SG_CfResL4', #Коэфф.рег.сопр.тр.ПГ4
#'YMBOR_COR', #конц бора
'YhqCor1_eqf', #Коэф рег смеш 1
'YhqCor2_eqf', #Коэф рег смеш 2
'YhqCor3_eqf', #Коэф рег смеш 3
#'asut02oint1', #Давленик в ГПК модели
#'cf_rlnvda',# Коэффициент температуры(?) в ПВД А
#'cf_rlnvdb'# Коэффициент температуры(?) в ПВД Б
]



u"""переменные, которые отдает модель"""
y_model=["YA11T810","YA12T811","YA13T812","YA14T813",#tgor
        "YA11T782","YA12T783","YA13T784","YA14T785",#thol
        "Y314B01","Y315B01","Y316B01","Y317B01",#dpgcn
        "Y065B022",#"Y062B01",#pzon
        s2g(u"14МЦ1-1"),s2g(u"14МЦ2-1"),s2g(u"14МЦ3-1"),s2g(u"14МЦ4-1"),#tpv
        s2g(u"14МЦ115Б"),s2g(u"14МЦ215Б"),s2g(u"14МЦ315Б"),s2g(u"14МЦ415Б"),#gpv
        s2g(u"14МЦ13"),s2g(u"14МЦ23"),s2g(u"14МЦ33"),s2g(u"14МЦ43")#ppg
        ]

u"""допустимые погрешности переменных"""
y_deriv=[0.01,0.01,0.01,0.01,
        0.005,0.005,0.005,0.005,
        0.005,0.005,0.005,0.005,
        0.005,
        0.005,0.005,0.005,0.005,
        0.005,0.005,0.005,0.005,
        0.005,0.005,0.005,0.005,
        ]


def obr_func(x,pos,v=v):
    u"""Функция, которую минимизируем для решения обр задачи
    x - вектор параметров
    pos -устарело- номер в архиве, на который натягиваем модель
    """
    global iresh
    global shod_sum
    global shod_vec
    global x0_obr_func
    print u"---------------"
    print u"Номер вызова функции:",iresh
    iresh+=1
    print u"вектор параметров - ",x
    print u"изменение вектора - ",x-x0_obr_func
    shod_vec.append(x)
    v.OPCCMD=8 #save(!) state
    x_bak=np.zeros(len(x_model))# сохраняем параметры
    print u"сохраняем параметры"
    for i in range(len(x_model)):
        if x_model[i]=='YHSIEVE_TUN':
            x_bak[i]=v[x_model[i]][0]
        else:
            x_bak[i]=v[x_model[i]]
    print u"вносим возмущение в параметры"
    for i in range(len(x_model)): # вносим возмущение в параметры
        if x_model[i]=='YHSIEVE_TUN':
            v[x_model[i]]=[x[i],1.,1.,1.,1.,1.,1.,1.,1.,1.]
        else:
            v[x_model[i]]=x[i]
    pr_dtdt()
    y_data=get_y_model()
    retmas=[]
    for i in range(len(y_model)):
        retmas.append(((v[y_model[i]]-y_data[i])/(y_deriv[i]*y_data[i])))
    retsumsq=0.0
    for i in retmas:
        retsumsq+=i**2
    shod_sum.append(retsumsq)
    ##Комментим блок, для лучшей сходимости
    v.OPCCMD=4 #load(!) state
    '''
    for i in range(len(x_model)): # возвращаем параметры
        if x_model[i]=='YHSIEVE_TUN':
            v[x_model[i]]=[x_bak[i],1.,1.,1.,1.,1.,1.,1.,1.,1.]
        else:
            v[x_model[i]]=x_bak[i]
    '''
    start_model()

    print u"Итоговая функция:",retsumsq
    print u"---------------"
    return retmas

def bound_obr_func(x,pos,v=v):
    u"""Функция граничных условий
    """
    global shod_sum
##    aa = []
##'''
##    for ix in x:
##        if ix<0.:
##            rr=np.zeros(len(y_model))
##            rr.fill(10.)
##            aa=rr
##            return aa
##    return obr_func(x,pos,v=v)
##'''
    if min(x)<=0:
        rr=np.zeros(len(y_model))
        rr.fill(10000.)
        aa=rr
        print u"Вышли за гран условия"
        shod_sum.append(len(y_model)*10000.*10000.)
    else:
        aa = obr_func(x,pos,v=v)
    return aa

#    return None

#sol3=obr_func([1.5,1.5,1.5,1.5,84.,84.,84.,84.,1.,0.01,0.01,0.01,0.01,2.3,2.3,2.3,2.3,7.7,0.,0.,0.,60.22,0.975,0.96],0)
#sol5=obr_func([1.5,1.5,1.5,1.5,84.,84.,84.,84.,1.,0.01,0.01,0.01,0.01,2.3,2.3,2.3,2.3,7.7,0.,0.,0.,60.22,0.975,0.12],0)
#sol4=bound_obr_func([1.5,1.5,1.5,1.5,84.,84.,84.,84.,1.,0.01,0.01,0.01,0.01,2.3,2.3,2.3,2.3,7.7,0.,0.,0.,60.22,0.975,0.96],0)

'''
x0_obr_func=[1.6,1.6,1.6,1.6,
            86.,86.,86.,86.,
            1.1,
            0.1,0.1,0.1,0.1,
            2.5,2.5,2.5,2.5,
            #8,
            0.1,0.1,0.1,
            #60.5,
            #0.95,
            #0.90]

x0_obr_func=[1.6,1.6,1.6,1.6,
            86.,86.,86.,86.,
            1.1,
            0.1,0.1,0.1,0.1,
            2.5,2.5,2.5,2.5,
            0.1,0.1,0.1,]

'''


'''
bounds=[]
for i in range(len(x0_obr_func)):
    bounds.append(lambda x,c: x[i])
bounds.append(lambda x,c: 1.3-x[22])
bounds.append(lambda x,c: 1.3-x[23])
bounds.append(lambda x,c: -0.5+x[22])
bounds.append(lambda x,c: -0.5+x[23])
boundmas=[[0,None],[0,None],[0,None],[0,None],
            [0,None],[0,None],[0,None],[0,None],
            [0,None],
            [0,None],[0,None],[0,None],[0,None],
            [0,None],[0,None],[0,None],[0,None],
            [0,None],
            [0,None],[0,None],[0,None],
            [0,None],
            [0.5,1.3],
            [0.5,1.3],
            ]
'''



def set_model_param_x(x,v=v):
    u"""Приводим модель к найденому решению
    """
    for i in range(len(x_model)):
        if x_model[i]=='YHSIEVE_TUN':
            v[x_model[i]]=[x[i],1.,1.,1.,1.,1.,1.,1.,1.,1.]
        else:
            v[x_model[i]]=x[i]
    pr_dtdt()
    v.reset_all=1

def rash1_set(v,value,value_set,param,step,statneed=True):
    u"""приводим переменную value к value_set, изменением param с шагом step
    пример rash1_set(v,"yafloop1_G",4300.,"Loop_CfResL1",1.0)"""
    growval=False
    degreeval=False
    while abs(v[value]-value_set)>10.:
        if v[value]>value_set:
            if degreeval:
                step=step/2
            v[param]+=step
            growval=True
        else:
            if growval:
                step=step/2
            v[param]-=step
            degreeval=True
        print step
        if(statneed):
            pr_dtdt()

def main_solve_obr_guess(Npg,tpvd1_r,tpvd2_r,pgpk):
    u"""основная программа решения обр задачи
    Npg - Мощность реактора (размерность базы СВРК)
    tpvd1_r - температура ПВД1 (размерность базы СВРК)
    tpvd2_r - температура ПВД2 (размерность базы СВРК)
    pgpk - давление в ГПК (размерность базы СВРК)"""
    global x0_obr_func
    global x_model
    global y_model
    global y_deriv
    global iresh
    global shod_vec
    global shod_sum
    global shodsum_mas
    global solve_obr_guess

    start = time.time()
    RelaxState()
    start_model_drive()
    start_model()
    v["OPCCMD_"]=112 #name for state
    ##gpetl_set([0.75,0.75,0.75,0.75],[4300.,4300.,4300.,4300.])
    ##sol=optimize.root(gpetl_set,[0.5,0.5,0.5,0.5],args=([4500.,4500.,4500.,4500.]))#,tol=0.01
    #solve_obr_guess=[] # массив решений задачи
    #shodsum_mas=[] #массив сходимости
    set_model_param_x(x0_obr_func)
    v.ASUT_RD1KEY+=1 #ставим РД-1 на турбине
    """model_perem=[set_ntp(Npg,[tpvd1_r,tpvd2_r],pgpk)]"""
    iresh=1
    shod_sum=[]
    shod_vec=[]
    x0_obr_func=np.zeros(len(x_model)) #Начальное состояние модели кладем в начальную точку решателя, для быстрой сходимости модели
    for i in range(len(x_model)):
        if x_model[i]=='YHSIEVE_TUN':
            x0_obr_func[i]=v[x_model[i]][0]
        else:
            x0_obr_func[i]=v[x_model[i]]
    pr_dtdt()

    diag_x=[#N positive entries that serve as a scale factors for the variables.
    1.,1.,1.,1.,
    1.,1.,1.,1.,
    1.,
    1.,1.,1.,1.,
    1.,1.,1.,1.,
    1.,1.,1.
    ]
    #Устанавливаем граничные условия:
    ##set_ntp(Npg/3000*100,[tpvd1_r,tpvd2_r],pgpk*10.197)
    sol=optimize.leastsq(bound_obr_func,x0_obr_func,args=(0),full_output=True,ftol=0.001,factor=10.,diag=None,epsfcn=0.1)#factor=10,tol=0.01
    #sol=optimize.curve_fit(bound_obr_func,p0=x0_obr_func,args=(0),full_output=True)
    ##sol=[0.,1.,2.,3.]
    ##sol=optimize.fmin_tnc(obr_func,x0_obr_func,args=([0]),approx_grad=True,bounds=boundmas,disp=5)
    ##sol=optimize.fmin_cobyla(obr_func,x0_obr_func,bounds,args=([0]),disp=3,maxfun=50)
    ##sol=optimize.fmin_slsqp(obr_func,x0_obr_func,args=([0]),disp=3,full_output=1)
    finish = time.time()
    print u"Скорость выполенения: ",(finish - start)/60.,u" минут"
    #solve_obr_guess.append(sol[0])
    #shodsum_mas.append(shod_sum)
    #set_model_param_x(sol[0])
    #show_inmfa(0)
    return sol,shod_sum,shod_vec


if __name__ == '__main__':
    print "!!!!!!!!!!!111111111"
    try:
        main_solve_obr_guess(v.OG_N_pg_calc/3000*100,v.OG_T_pvd1,v.OG_T_pvd2,v.OG_P_gpk)
    except:
        print "1",sys.exc_info()
        print "1"
        time.sleep(5)
        pass
    time.sleep(5)
    raw_input()




#!/usr/bin/env python
# -*- coding: utf-8 -*-

from knpp_b3_hdf5_model import *
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from def_vars import *
import os

rc('font', **{'family': 'verdana', 'size'   : 12})
rc('text.latex', unicode=True)
rc('text.latex', preamble='\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\usepackage[russian]{babel}')

locale.setlocale(locale.LC_ALL,'rus')

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
u'YhqCor3_eqf',
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
u'YhqCor1_eqf':ur'$k_{смеш1}$',
u'YhqCor2_eqf':ur'$k_{смеш2}$',
u'YhqCor3_eqf':ur'$k_{смеш3}$',
u'Nin':ur'$N_{АЗ}$',
u'Pazin':ur'$P_{АЗ}$',
u'Pgpkin':ur'$P_{гпк}$',
u'Tpvdain':ur'$t_{пвд1}$',
u'Tpvdbin':ur'$t_{пвд2}$'
}

mod_coef_delt=[
0.1,0.1,0.1,0.1,
1,1,1,1,
0.1,0.1,0.1,0.1,
0.1,0.1,0.1,0.1,
0.1,0.1,0.1]

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
'Pzone2',
'Ntep','Naz','Nrr','N1k','N2k','Naknp','Nturb',
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
'Ppg4',
'Pgpk',
'tpvd1','tpvd2',
'ppvd1','ppvd2',
'gpvd1','gpvd2',
'ppv1','ppv2','ppv3','ppv4',
'gkgtn'
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


coef_bound=np.array([
[0.,5.,10.],
[0.1],
[0.1],
[0.1],
[50.,75.,100.],
[84.5],
[84.5],
[84.5],
[0.,3.5,7.],
[0.,3.5,7.],
[0.3],
[0.3],
[0.3],
[0.,20.,40.],
[1.5],
[1.5],
[1.5],
[0.,5.,10.],
[0.,5.,10.],
[0.,5.,10.],
[94.,100.,104.], #N
[59.,60.,61.5], #Pgpk
[215.], #Tpvd1
[215.], #Tpvd2
[160.] #Paz
])


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


if os.environ['COMPUTERNAME']=='MOLEV':
    dirofdis='D:/git_py/' #work
else:
    dirofdis='G:/git_disser/disser/' #home

'''old-good 5 poi?
storeofd = pd.HDFStore(dirofdis+'liner_JAC_model_x0_by1.h5')
'''
'''new *10 points'''
storeofd = pd.HDFStore(dirofdis+'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0_10points.h5')

#dataindf=pd.DataFrame({'Nin':storeofd['Nin'],'pgpkin':storeofd['pgpkin'],'tpvd1in':storeofd['tpvd1in'],'tpvd2in':storeofd['tpvd2in'],'pazin':storeofd['pazin']})
#storeofd['Nin']=pd.Series(optim_npmas.transpose()[1])
#storeofd['pgpkin']=pd.Series(optim_npmas.transpose()[2])
#storeofd['tpvd1in']=pd.Series(optim_npmas.transpose()[3])
#storeofd['tpvd2in']=pd.Series(optim_npmas.transpose()[4])
#storeofd['pazin']=pd.Series(optim_npmas.transpose()[5])
#storeofd['pmodelout']=pd.Series(optim_npmas.transpose()[6])
out_data=storeofd['model_data']
inp_data=storeofd['inp_data']
storeofd.close()


y0=out_data.iloc[0]
y0m=np.array([out_data.iloc[0][x] for x in yexpvar])
x0=inp_data.iloc[0]
x0m=np.array([inp_data.iloc[0][x] for x in mod_coef])

##qq=inp_data[
##(inp_data['FLaSG2_CfResL']==0.1)&
##(inp_data['FLaSG3_CfResL']==0.1)&
##(inp_data['FLaSG4_CfResL']==0.1)&
##(inp_data['YD11D01_2_Hnom']==0.1)
##]['FLaSG1_CfResL']
##
##
##qq=inp_data[inp_data['FLaSG2_CfResL']==0.1]
##
##qq=qq[inp_data['FLaSG3_CfResL']==0.1]
##
##qq=inp_data
##for k in qq.keys()[1:]:
##    print k,"\n",qq
##    qq=qq[qq[k]==qq[qq[k].duplicated()][k].values[0]]
##qq.index
##
##plt.plot(qq['FLaSG1_CfResL'],out_data.ix[qq.index]['Tgor1'],'o-',label='Tgor1');plt.show()



def all_graf(vkey):
    u"""в графиках показываем откланения искомых переменных от возмущений переменных модели
    """
    qq=inp_data
    u'''старый непонятный выбор изменяемых элементов
    for k in qq.keys().drop(vkey):
        #print k,"\n",qq
        qq=qq[qq[k]==qq[qq[k].duplicated()][k].values[0]]
    qq.index
    '''
    qqn=qq[qq[vkey]!=qq[vkey][0]] #выбрали только изменения переменной vkey в архиве
    #u""" РАССМОТРЕТЬ! ВОЗМОЖНО ОСТАВИТЬ ЗАКОММЕНЧЕННУЮ!
    for kk in yexpvar : #out_data.ix[qq.index].keys()
        plt.plot(qq[vkey],out_data.ix[qq.index][kk],'o')
        plt.plot(qqn[vkey],out_data.ix[qqn.index][kk],'r-')
        plt.xlabel(mod_coef_lable[vkey])
        plt.ylabel(kk+'\t'+str(deriv_test[kk]))
        if abs(plt.ylim()[1]-plt.ylim()[0])<deriv_test[kk]:
            print kk,'\tout of limits'
            plt.ylim((out_data.ix[qqn.index][kk].mean()-deriv_test[kk],out_data.ix[qqn.index][kk].mean()+deriv_test[kk]))
        print kk,'\t',out_data.ix[qqn.index][kk].max()-out_data.ix[qqn.index][kk].min(),'\t',out_data.ix[qqn.index][kk].max()-out_data.ix[qqn.index][kk].min(),'\t',deriv_test[kk]


        deriv1old=[(out_data.ix[qqn.index][kk].iloc[3]-out_data.ix[qqn.index][kk].iloc[0])/(qqn[vkey].iloc[3]-qqn[vkey].iloc[0])]
        deriv1=[(out_data.ix[qqn.index][kk].iloc[3]-y0[kk])/(qqn[vkey].iloc[3]-x0[vkey])]
        print '\t',deriv1
        for iii in range(4):
            #deriv1.append((out_data.ix[qqn.index][kk].iloc[iii+1]-out_data.ix[qqn.index][kk].iloc[iii])/(qqn[vkey].iloc[iii+1]-qqn[vkey].iloc[iii]))
            nud=(out_data.ix[qqn.index][kk].iloc[iii]-y0[kk])/(qqn[vkey].iloc[iii]-x0[vkey])
            deriv1.append(nud)
            print nud,
        derivnp=np.array(deriv1)
        print 'mean= ',derivnp.mean(),derivnp.std()

        plt.show()
    """
    for kk in out_data.ix[qq.index].keys():
        plt.plot(qqn[vkey],out_data.ix[qqn.index][kk],'o-')
        plt.xlabel(vkey)
        plt.ylabel(kk)
        plt.show()
    """
outlim1=[]
outlim2=[]


def JAC_from_arch(vkey,pr_opt=False):
    u"""
    ищем один столбец якобиана из архива
    наверное для теста
    """
    qq=inp_data
    u'''старый непонятный выбор изменяемых элементов
    for k in qq.keys().drop(vkey):
        #print k,"\n",qq
        qq=qq[qq[k]==qq[qq[k].duplicated()][k].values[0]]
    print qq.index
    '''
    qqn=qq[qq[vkey]!=qq[vkey][0]] #выбрали только изменения переменной vkey в архиве
    der={}
    derfit={}
    x0fit={}
    for kk in yexpvar: #out_data.ix[qq.index].keys()
        if abs(out_data.ix[qqn.index][kk].max()-out_data.ix[qqn.index][kk].min())<0.1*arch_var_deviation[kk]:
            u'''если на всем интервале изменения параметра модели
             выходной параметр меньше десятой доли погрешности,
              то обнуляем якобиан
              раньше везде вместо arch_var_deviation был deriv_test'''
            if pr_opt: print u'index\tmax-min из всех знач\tmaxn-minn из якобиантн знач\tконстанта отсечения погрешности'
            if pr_opt: print kk,'\t',out_data.ix[qq.index][kk].max()-out_data.ix[qq.index][kk].min(),'\t',out_data.ix[qqn.index][kk].max()-out_data.ix[qqn.index][kk].min(),'\t',arch_var_deviation[kk]
            if pr_opt: print kk,'\tout of limits'
            der[kk]=0
            outlim1.append(vkey+'\t'+kk)
        else:
            if pr_opt: print u'index\tmax-min из всех знач\tmaxn-minn из якобиантн знач\tконстанта отсечения погрешности'
            if pr_opt: print kk,'\t',out_data.ix[qq.index][kk].max()-out_data.ix[qq.index][kk].min(),'\t',out_data.ix[qqn.index][kk].max()-out_data.ix[qqn.index][kk].min(),'\t',arch_var_deviation[kk]
            #производная по максимальным точкам:
            deriv1old=[(out_data.ix[qqn.index][kk].iloc[qqn[vkey].shape[0]-1]-out_data.ix[qqn.index][kk].iloc[0])/(qqn[vkey].iloc[qqn[vkey].shape[0]-1]-qqn[vkey].iloc[0])]
            deriv1=[(out_data.ix[qqn.index][kk].iloc[qqn[vkey].shape[0]-1]-y0[kk])/(qqn[vkey].iloc[qqn[vkey].shape[0]-1]-x0[vkey])]
            if pr_opt: print '\t',deriv1
            for iii in range(qqn[vkey].shape[0]-1):
                #производная по маленьким отрезкам
                #deriv1.append((out_data.ix[qqn.index][kk].iloc[iii+1]-out_data.ix[qqn.index][kk].iloc[iii])/(qqn[vkey].iloc[iii+1]-qqn[vkey].iloc[iii]))
                nud=(out_data.ix[qqn.index][kk].iloc[iii+1]-y0[kk])/(qqn[vkey].iloc[iii+1]-x0[vkey])
                deriv1.append(nud)
                if pr_opt: print nud,
            derivnp=np.array(deriv1)
            if pr_opt: print 'mean= ',derivnp.mean(),derivnp.std(),
            der[kk]=derivnp.mean()
            if abs(derivnp.mean())<2*derivnp.std():
                if pr_opt: print "!!!bad deriv",
                der[kk]=0
                outlim2.append(vkey+'\t'+kk)
        #fit
        yfit=out_data.ix[qqn.index][kk].values
        xfit=qqn[vkey].values
        def ffit(x,a,b):
            return a*x+b
        solvefit,solvefitcov=scipy.optimize.curve_fit(ffit,xfit,yfit)
        yfplt=ffit(xfit,*solvefit)
        derfit[kk]=solvefit[0]
        x0fit[kk]=solvefit[1]
        if pr_opt:print kk, der[kk], derfit[kk], x0fit[kk]

        u"""секция вывода на графики
        #fig = plt.figure()
        plt.plot(xfit,yfplt,'g-')
        plt.plot(qq[vkey],out_data.ix[qq.index][kk],'o')
        #plt.plot(qqn[vkey],out_data.ix[qqn.index][kk], 'r+',yerr=3.0)
        plt.errorbar(qqn[vkey],out_data.ix[qqn.index][kk], yerr=arch_var_deviation[kk]/2.,fmt='o',color='red')
        plt.xlabel(mod_coef_lable[vkey],fontsize=16)
        plt.ylabel(yexpvar_lable[kk],fontsize=16)
        #plt.ylabel(kk+'   deriv='+str(arch_var_deviation[kk]))
        if abs(plt.ylim()[1]-plt.ylim()[0])<arch_var_deviation[kk]:
            print kk,'\tout of limits'
            plt.ylim((out_data.ix[qqn.index][kk].mean()-arch_var_deviation[kk],out_data.ix[qqn.index][kk].mean()+arch_var_deviation[kk]))
        plt.show()
        #fig.savefig(dirofdis+'plt2/'+kk+'_'+vkey+'.png')
        #"""
        if pr_opt: print
        if pr_opt: print
    return derfit,x0fit

def all_JAC(pr_opt=False):
    jac={}
    x0masfit={}
    for jj in mod_coef: #inp_data.keys().drop(u'YHSIEVE_TUN')
        if pr_opt: print jj
        jac[jj],x0masfit[jj]=JAC_from_arch(jj)
    return jac,x0masfit

def jac_2_matrix(jac,pr_opt=False):
    mas=[]
    for jj in mod_coef: #inp_data.keys().drop(u'YHSIEVE_TUN')
        masinp=[]
        for ij in yexpvar: #out_data.ix[qq.index].keys()
            masinp.append(jac[jj][ij])
            if pr_opt: print jj,ij,jac[jj][ij]
        mas.append(masinp)
        if pr_opt: print jj,masinp
    mas=np.array(mas)
    return mas  # mas - матрица якобиана в виде np.array, строки, столбцы -yexpvar,mod_coef

jac,x0masfit=all_JAC()
mas=jac_2_matrix(jac)
masx0=jac_2_matrix(x0masfit)

def search_changes_jac():
    u"""
    ищем изменения якобиана в зависимости от начальной точки
    """
    files_arr=['liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0_10points.h5',
                r'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 159.0.h5',
                r'liner_JAC_model_x098.0, 60.9, 215.0, 215.0, 160.0.h5',
                r'liner_JAC_model_x0100.0, 60.9, 220, 215.0, 160.0.h5',
                r'liner_JAC_model_x0100.0, 61.3, 215, 215.0, 160.0.h5',
                r'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0FLaSG1_CfResL_2.0_10.h5',
                r'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0YD11D01_2_Hnom_110.0_10.h5',
                r'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0Loop_CfResL1_1.5_10.h5',
                r'liner_JAC_model_x0100.0, 60.9, 215.0, 215.0, 160.0SG_CfResL1_2.5_10.h5']
    arr_names=['beg',
                'p159',
                'n98',
                't220',
                'p2_61_3',
                'FLaSG1_CfResL',
                'YD11D01_2_Hnom',
                'Loop_CfResL1',
                'SG_CfResL1'                ]
    jacdict={}
    for i,fl in enumerate(files_arr):
        storeofd = pd.HDFStore(dirofdis+fl)
        out_data=storeofd['model_data']
        inp_data=storeofd['inp_data']
        storeofd.close()
        y0=out_data.iloc[0]
        y0m=np.array([out_data.iloc[0][x] for x in yexpvar])
        x0=inp_data.iloc[0]
        x0m=np.array([inp_data.iloc[0][x] for x in mod_coef])
        jac,x0masfit=all_JAC()
        jdf=pd.DataFrame(jac)
##        jdfxnorm=jdf.copy()
##        for par in mod_coef:
##            dmt=mod_coef_delta[par][1]-mod_coef_delta[par][0] #временная переменная для нормировки
##            jdfxnorm[par]=jdfxnorm[par]
##        normcoef_mas=np.array([normcoef[p] for p in mod_coef])
##        dx_norm=dx/normcoef_mas
        jacdict[arr_names[i]]=jdf
    jacpnl=pd.Panel(jacdict)
    #show smth
    #search
    for ex in yexpvar:
        print ex
        pd.options.display.float_format = '{:,.3f}'.format
        pd.set_option('expand_frame_repr', True)
        cond=abs(jacpnl.major_xs(ex).T.mean()*ncdf/arch_var_deviation[ex])>0.1
        print ((jacpnl.major_xs(ex).T*ncdf/arch_var_deviation[ex]).T)[cond] #normirovanniy jac
        print
        print
    #old
    jacpnl.major_xs(u'Tgor1')[cond].T.std()*100./jacpnl.major_xs(u'Tgor1')[cond].T.mean()
    normcoef={}
    for par in mod_coef:
        normcoef[par]=mod_coef_delta[par][1]-mod_coef_delta[par][0]
    ncdf=pd.Series(normcoef)
    jacpnl.major_xs(u'Tgor1').T.mean()*ncdf/arch_var_deviation[u'Tgor1']
    cond=abs(jacpnl.major_xs(u'Tgor1').T.mean()*ncdf/arch_var_deviation[u'Tgor1'])>0.1
    jacpnl.major_xs(u'Tgor1').T.std()*ncdf/arch_var_deviation[u'Tgor1']



def test():
    u"""Проверили правильность якобиана, все ок
    """
    print inp_data.keys().drop(u'YHSIEVE_TUN').shape #вектор изменяемых параметров X 19
    print out_data.ix[qq.index].keys().shape #вектор получаемых значений Y 45
    x1=44
    x2=44
    xx1= np.array([inp_data.iloc[x1][x] for x in mod_coef]) #начальные значения вектора
    yy1= np.array([out_data.iloc[x1][x] for x in yexpvar]) #соответствующие значения функционала
    xx2= inp_data.drop(u'YHSIEVE_TUN',axis=1).iloc[x2] #конечные значения вектора
    yy2= out_data.iloc[x2] #соответствующие значения функционала
    dx=xx1-x0m
    print dx
    dy=np.dot(mas.T,dx.T)
    print dy
    #dx=dx.values
    #itog s jac
    #yj2=yy1+np.dot(mas.T,dxx.T).T[0]
    yj1=y0m+dy
    res=yy1-yj1
    for i,rr in enumerate(yexpvar):
        print rr,'\t', yy1[i],'\t',yj1[i],'\t',res[i],'\t',res[i]*100/yy1[i]

    for j in inp_data.keys().drop(u'YHSIEVE_TUN'):
        print j
        for ij in jac[j]:
            print '\t',ij,'\t',jac[j][ij]

    '''write JAC to
    '''
    jacdf=pd.DataFrame(jac)
    writer=pd.ExcelWriter(dirofdis+'jac.xlsx')
    jacdf.to_excel(dirofdis+'jac.xlsx',sheet_name='JAC')
    witer.save()
    jacdf.to_html(dirofdis+'jac.html',float_format=lambda x: '%10.2f' % x)#,float_format='%5.2f'




def funk_fr_jac(dx):
    bounddx=[5,5,5,5,
    5,5,5,5,
    5,5,5,5,
    5,5,5,5,
    #5,
    5,5,5]
    for i,x in enumerate(dx):
        if abs(x)>bounddx[i]:
            aa=np.empty(25)
            aa.fill(10000)
            return aa
    return (yexp1-y0m-np.dot(mas.T,dx))#/yexp1 #разница между значением функции и постчитанным значением

u"""
для нормировки без зашкала нужно прийти к нулевой точке отсчета
"""
x00=[]
for par in mod_coef:
    x00.append(mod_coef_delta[par][0])
y00=y0m+np.dot(mas.T,x00-x0m)
x00=np.array(x00)
y00=np.array(y00)

def dx2x(dx):
    u"""
    Функция перевода изменения Х в значение Х от Х0
    нужна для нормальной работы минимизатора
    """
    #x=x0m+dx
    x=x00+dx
    return x

def x2dx(x):
    u"""
    ОБРАТНАЯ Функция перевода изменения DХ в значение Х от Х0
    нужна для нормальной работы минимизатора
    """
    #dx=x-x0m
    dx=x-x00
    return dx

def normolizeX(dx):
    u"""
    нормализуем отклонение параметров модели
    """
    normcoef={}
    for par in mod_coef:
        normcoef[par]=mod_coef_delta[par][1]-mod_coef_delta[par][0]
    normcoef_mas=np.array([normcoef[p] for p in mod_coef])
    dx_norm=dx/normcoef_mas
    return dx_norm

def normolizeX_ob(dx_norm):
    u"""
    ОБРАТНО возвращаем отклонение параметров модели от нормализованных
    """
    normcoef={}
    for par in mod_coef:
        normcoef[par]=mod_coef_delta[par][1]-mod_coef_delta[par][0]
    normcoef_mas=np.array([normcoef[p] for p in mod_coef])
    dx=dx_norm*normcoef_mas
    return dx


def boundX(x,fullprint=True):
    u"""
    ограничиваем Х
    """
    rel_err=np.zeros(x.shape)
    nx=x
    for i,xx in enumerate(nx):
        if xx<mod_coef_delta_m[i][0]:
            nx[i]=mod_coef_delta_m[i][0]
            if fullprint:
                print 'out of bound',i,xx,r'<',mod_coef_delta_m[i][0]
            rel_err[i]=mod_coef_delta_m[i][0]-xx
        elif xx>mod_coef_delta_m[i][1]:
            nx[i]=mod_coef_delta_m[i][1]
            if fullprint:
                print 'out of bound',i,xx,r'>',mod_coef_delta_m[i][1]
            rel_err[i]=xx-mod_coef_delta_m[i][0]
    return nx,rel_err

def y_fr_model():
    u"""
    Берем экспериментальные данные из архива станции
    """
    ppgpravk=0
    arch_var=dict(Tgor1=v.OG_T_gor[0],Tgor2=v.OG_T_gor[1],Tgor3=v.OG_T_gor[2],Tgor4=v.OG_T_gor[3],
        Thol1=v.OG_T_hol[0],Thol2=v.OG_T_hol[1],Thol3=v.OG_T_hol[2],Thol4=v.OG_T_hol[3],
        dPgcn1=v.OG_pp_gcn[0],dPgcn2=v.OG_pp_gcn[1],dPgcn3=v.OG_pp_gcn[2],dPgcn4=v.OG_pp_gcn[3],
        Pzone1=v.OG_p_rea,Pzone2=0.,
        Ntep=0.,Naz=0.,Nrr=0.,N1k=v.OG_N_1k,N2k=v.OG_N_pg_calc,Naknp=v.OG_N_aknp,Nturb=v.OG_N_gen,
        Tpv1=v.OG_t_pitv[0],Tpv2=v.OG_t_pitv[1],Tpv3=v.OG_t_pitv[2],Tpv4=v.OG_t_pitv[3],
        Gpv1=v.OG_g_pitv[0],Gpv2=v.OG_g_pitv[1],Gpv3=v.OG_g_pitv[2],Gpv4=v.OG_g_pitv[3],
        Ppg1=v.OG_p_pg[0]+ppgpravk,Ppg2=v.OG_p_pg[1]+ppgpravk,Ppg3=v.OG_p_pg[2]+ppgpravk,Ppg4=v.OG_p_pg[3]+ppgpravk,
        Pgpk=v.OG_P_gpk,
        tpvd1=v.OG_T_pvd1,tpvd2=v.OG_T_pvd2,
        ppvd1=v.OG_P_pvd[0],ppvd2=v.OG_P_pvd[1],
        gpvd1=v.OG_G_pvd[0],gpvd2=v.OG_G_pvd[1],
        ppv1=v.OG_p_pitvg[0],ppv2=v.OG_p_pitvg[1],ppv3=v.OG_p_pitvg[2],ppv4=v.OG_p_pitvg[3],
        gkgtn=0.)
    yexp1=np.array([arch_var[x] for x in yexpvar])
    return yexp1

y_from_model_mas=y_fr_model() # для ускорения считаем только 1 раз

def Ppg_popravka(ppgpravk):
    u"""
    Вводим аддитивную поправку на давления парогенераторов.
    !Учесть, что при замене массива параметров номер в массиве поменяется!
    """
    global y_from_model_mas
    y_from_model_mas_prav=y_from_model_mas.copy()
    y_from_model_mas_prav[32]+=ppgpravk
    y_from_model_mas_prav[31]+=ppgpravk
    y_from_model_mas_prav[30]+=ppgpravk
    y_from_model_mas_prav[29]+=ppgpravk
    return y_from_model_mas_prav




def f4minimise(x,xaddppg):
    u"""
    Функция для минимизации в методе наименьших квадратов
    минимизирует разницу между между экспериментальным значением датчиков и датчиками модели, настраивыми по изменению параметров модели
    минимизируем (yexp-y(x))**2->min
    или (yexp-y0-JAC*(x-x0))**2->min
    dx=x-x0 - входной параметр
    x0 и y0 некоторые начальные значения из модели (по ним строили якобиан)
    """
    yexp1=Ppg_popravka(xaddppg)
    yexperr=np.array([arch_var_deviation[ee] for ee in yexpvar])
    #minimize_polinomial=(yexp1-y0m-np.dot(mas.T,x-x0m))/yexperr
    minimize_polinomial=(yexp1-y00-np.dot(mas.T,x-x00))/yexperr
    return minimize_polinomial

def f4minimise_buf(dxnorm,fullprint=True):
    u"""
    первая буферная функция для минимизации которая берет dxnorm а не x
    """
    global shag
    global s_sum
    if fullprint:
        print '_________________________________________________________________'
        print u"ШАГ ",shag
        print 'vector dxnorm-',dxnorm
    dx=normolizeX_ob(dxnorm[:-1]) #перевели в ненормированный вид
    xaddppg=dxnorm[-1]*4.#последний с конца элемент - аддитивная поправка давлений на выходе из ПГ)
    if fullprint:
        print 'vector dx-',dx
    x=dx2x(dx) #перевели от изменений к Х
    if fullprint:
        print 'vector x-',x
    xbounded,rel_x=boundX(x,fullprint) #отсекли изменения сверх предела
    reln_x=normolizeX(rel_x) #Нормализуем отклонение от границ параметров
    k_bounded=1+reln_x.sum()/1000. #множитель увеличивающийся с отклонением от границ параметров
    if fullprint:
        print u"множитель к баунд=",k_bounded
        print u"поправка к P ПГ=",xaddppg
    minimize_polinomial=f4minimise(xbounded,xaddppg)*k_bounded#*k_bounded
    shag+=1
    #вводим поправку на отклонение коэффициентов друг от друга
    #xotklmas=[x[0]-x[1],x[0]-x[2],x[0]-x[3]]
    #minimize_polinomial=np.append(minimize_polinomial,xotklmas)
    if fullprint:
        print "result of min func = ",minimize_polinomial
    s_sum_t=0 #сумма квадратов текущего вывода функции отклонения
    for e in minimize_polinomial: s_sum_t+=e*e
    s_sum.append(s_sum_t) #массив общего расчета вывода функций отклонения
    if fullprint:
        print "sum of sqr func= ",s_sum_t
    return minimize_polinomial


def restrict_JAC(JT):
    u"""
    добавляем к матрице Якоби члены для ограничения коэффициентов (по удалению их друг от друга)
    также добавляем поправку к давлениям ПГ
    """
    normcoef={}
    for par in mod_coef:
        normcoef[par]=mod_coef_delta[par][1]-mod_coef_delta[par][0]
    normcoef_mas=np.array([normcoef[p] for p in mod_coef])
    Jypsev1=np.zeros(24) #FLaSG1_CfResL-FLaSG2_CfResL
    Jypsev1[0]=1./(0.2*normcoef_mas[0]) #dypsev/dx1
    Jypsev1[1]=-1./(0.2*normcoef_mas[0])

    Jypsev2=np.zeros(24) #FLaSG1_CfResL-FLaSG3_CfResL
    Jypsev2[0]=1./(0.2*normcoef_mas[0]) #dypsev/dx1
    Jypsev2[2]=-1./(0.2*normcoef_mas[0])
    Jypsev=np.concatenate(([Jypsev1],[Jypsev2]))

    Jypsev3=np.zeros(24) #FLaSG1_CfResL-FLaSG4_CfResL
    Jypsev3[0]=1./(0.2*normcoef_mas[0]) #dypsev/dx1
    Jypsev3[3]=-1./(0.2*normcoef_mas[0])
    Jypsev=np.concatenate((Jypsev,[Jypsev3]))

    Jypsev4=np.zeros(24) #Loop_CfResL1-Loop_CfResL2
    Jypsev4[8]=1./(0.2*normcoef_mas[8]) #dypsev/dx1
    Jypsev4[9]=-1./(0.2*normcoef_mas[8])
    Jypsev=np.concatenate((Jypsev,[Jypsev4]))

    Jypsev5=np.zeros(24) #Loop_CfResL1-Loop_CfResL3
    Jypsev5[8]=1./(0.2*normcoef_mas[8]) #dypsev/dx1
    Jypsev5[10]=-1./(0.2*normcoef_mas[8])
    Jypsev=np.concatenate((Jypsev,[Jypsev5]))

    Jypsev6=np.zeros(24) #Loop_CfResL1-Loop_CfResL4
    Jypsev6[8]=1./(0.2*normcoef_mas[8]) #dypsev/dx1
    Jypsev6[11]=-1./(0.2*normcoef_mas[8])
    Jypsev=np.concatenate((Jypsev,[Jypsev6]))

    Jypsev7=np.zeros(24) #SG_CfResL1-SG_CfResL2
    Jypsev7[12]=1./(0.2*normcoef_mas[12]) #dypsev/dx1
    Jypsev7[13]=-1./(0.2*normcoef_mas[12])
    Jypsev=np.concatenate((Jypsev,[Jypsev7]))

    Jypsev8=np.zeros(24) #SG_CfResL1-SG_CfResL2
    Jypsev8[12]=1./(0.2*normcoef_mas[12]) #dypsev/dx1
    Jypsev8[14]=-1./(0.2*normcoef_mas[12])
    Jypsev=np.concatenate((Jypsev,[Jypsev8]))

    Jypsev9=np.zeros(24) #SG_CfResL1-SG_CfResL2
    Jypsev9[12]=1./(0.2*normcoef_mas[12]) #dypsev/dx1
    Jypsev9[15]=-1./(0.2*normcoef_mas[12])
    Jypsev=np.concatenate((Jypsev,[Jypsev9]))

    Jn=np.concatenate((JT.T,Jypsev))
    JTrestr=(Jn.T).copy()

    #add PG poprav:
    pgadd=np.zeros(54)
    pgadd[29]=1.
    pgadd[30]=1.
    pgadd[31]=1.
    pgadd[32]=1.
    JTpga=np.concatenate((JTrestr,[pgadd]))
    JTrestr=JTpga
    return JTrestr

def restrict_y0(y0,JTrestr):
    u"""
    y0 - y0m - начальная точка
    JTrestr - новый якобиан с ограничениями
    приводим начальную точку модели к размерности нового ограниченного якобиана
    """
    y00add=np.zeros(JTrestr.shape[1]-y0.shape[0]) #ставим все нули!
    y0restr=(np.append(y0,y00add)).copy()
    return y0restr

def restrict_yexp(yexp,JTrestr):
    u"""
    yexp - yexp1 - экспериментальный срез из блока
    JTrestr - новый якобиан с ограничениями
    приводим начальную точку модели к размерности нового ограниченного якобиана
    """
    yexpadd=np.zeros(JTrestr.shape[1]-yexp.shape[0]) #ставим все нули!
    yexprestr=(np.append(yexp,yexpadd)).copy()
    return yexprestr

def restrict_yexperr(yexperr,JTrestr):
    u"""
    yexperr - yexperr - позволительная ошибка новых данных
    JTrestr - новый якобиан с ограничениями
    приводим начальную точку модели к размерности нового ограниченного якобиана
    """
    yexperradd=np.ones(JTrestr.shape[1]-yexperr.shape[0]) #perepravit!
    yexperr_restr=(np.append(yexperr,yexperradd*2)).copy()
    return yexperr_restr

def search_sum(delta):
    u"""
    возвращаем сумму квадратов по дельте
    """
    return (((-yexp1+y0m)/yexperr+np.dot(JT.T,delta))**2).sum() #

def func_by_delt(delta,JTnenorm,y0m):
    u"""
    возвращаем значения модели по параметрам
    """
    yf=y0m+np.dot(JTnenorm.T,delta)
    return yf

def otchet_func(delta):
    u"""
    даем полный отчет по функции по решению задачи
    """
    print "_________"
    print "_________"
    np.set_printoptions(precision=4,suppress=True)
    for pmc in mod_coef:
        print pmc+'\t',
    print
    print u"Дельта найденных коэффициентов"
    print delta
    print u"Найденные коэффициенты"
    print np.append(x0m,0.)+delta
    print "_________"
    print u"функция"
    for pmc in yexpvar:
        print pmc+'\t',
    print
    yf=y0m+np.dot(JTnenorm.T,delta)
    print yf
    print u"разница с экспериментом"
    yfd=y0m+np.dot(JTnenorm.T,delta)-yexp1
    print yfd
    print u"exp + разница с экспериментом построчно +err"
    print
    for i,pmc in enumerate(yexpvar):
        print i,'\t',
        print pmc,'\t',
        print yf[i],'\t',
        print yfd[i],'\t',
        print yexperr[i],'\t',
        print
    print "_________"
    print "_________"
    print u"сумма квадратов"
    print search_sum(delta)
    print "_________"
    print "_________"

def newton_gauss(JTnorm,yexp1,y0m,yexperr,fullprint=False):
    u"""ищем решение методом ньютона гауса
    JTnorm - нормированная(!!!) транспонированная матрица Якоби
    yexp1 - экспериментальный срез
    y0m - нулевая точка модели, от которой считаем функцию
    yexperr - массив ошибок
    f(x0+dd)=f(x0)+Jac*dd
    Smin(x0+dd)=normvec(y-f(x0)-Jac**dd)**2
    take derive
    (JacT*Jac)dd=JacT(y-f(x0))
    dd=(JacT*Jac)**-1*JacT(y-f(x0))
    """
    #jac=all_JAC()
    u"""linear search solve
    jdfn['Nin']
    """
    #global y0m
    #new psevd datch
    #ypsev1=(x1-x2)/(0.2*normcoef_mas[i])

    #1
    #Jypsev.append(np.zeros(24)) #FLaSG1_CfResL-FLaSG2_CfResL
    np.set_printoptions(precision=4,suppress=True)

    #yexp1=Ppg_popravka(0)
    #yexperr=np.array([arch_var_deviation[ee] for ee in yexpvar])
    #jdf=pd.DataFrame(jac)
    #jdfn=(jdf.T/pd.Series(arch_var_deviation)).T
    #JTnenorm=jac_2_matrix(jdf)  #NEnormir jac
    #JT=jac_2_matrix(jdfn) #normir jac
    #Jnn=np.concatenate((JTnenorm.T,Jypsev))
    #JTnenorm=Jnn.T
    #JTpganenorm=np.concatenate((JTnenorm,[pgadd]))
    #JTnenorm=JTpganenorm

    JTJ=np.dot(JTnorm,JTnorm.T)
    JTJ1=np.linalg.inv(JTJ)

    dsal=np.dot((np.linalg.inv(np.dot(JTnorm,JTnorm.T))) , np.dot(JTnorm,(yexp1-y0m)/yexperr)) #to4noe reshenie
    if fullprint: otchet_func(dsal)

    def tfunc(delta,JTnorm,yexp1,y0m):
        #print delta
        #summ.append(search_sum(delta))
        return ((-yexp1+y0m)/yexperr+np.dot(JTnorm.T,delta))
    def tfunc_jac(delta,JTnorm):
        return JTnorm.T
    xprib0=np.zeros(25) #начальное приближение решения
    xprib0.fill(0.1)
    xprib0[20]=-0.1

    b1=(np.append(mod_coef_delta_m.T[0]-x0m,-np.inf),np.append(mod_coef_delta_m.T[1]-x0m,np.inf))
    summ=[]
    dsalscp=scipy.optimize.least_squares(tfunc,xprib0,bounds=b1, args=(JTnorm,yexp1,y0m))#jac=tfunc_jac,max_nfev =1000,diff_step=0.1
#    plt.plot(summ)
#    plt.xlabel(u'Шаг решения')
#    plt.ylabel(u'Сумма квадратов нормы вектор-функции')
#    plt.show()

    if fullprint: otchet_func(dsalscp['x'])
    memory1=dsalscp['x']
    memory2=dsalscp['x']
    memory3=dsalscp['x']

    '''dds=np.dot((np.dot(np.linalg.inv(np.dot(mas,mas.T)),mas)),(yexp1-y0m)/yexperr)

    dds=np.dot(JTJ1,np.dot(JT,(yexp1-y0m-np.dot(JT.T,np.array([0.1,0.1,0.1,0.1,14,14,14,14,0,0,0,0,0,0,0,0,0,0,0,8,2,1,10,10])))/yexperr))
    mnojt=np.dot(JTJ1,JT)
    ddlm=np.dot(JTJ1,JT)
    ##dds=np.dot((np.dot(np.linalg.inv(np.dot(mas,mas.T)),mas)),(yexp1-y00))
    ##dds2=np.dot((np.dot(np.linalg.inv(np.dot(mas,mas.T)),mas)),(yexp1-y00-np.dot(mas.T,dds)))
    ##dds1=np.dot((np.dot(np.linalg.inv(np.dot(mas,mas.T)),mas)),(yexp1-y00)/yexperr)
    dds0=dds.fill(0.0)
    dds0=dds
    for inw in range(10):
        #dds1=np.dot((np.dot(np.linalg.inv(np.dot(mas,mas.T)),mas)),(yexp1-y00-np.dot(mas.T,dds0))/yexperr)
        dds1=np.dot(np.linalg.inv(np.dot(mas,mas.T)),np.dot(mas,(yexp1-y00-np.dot(mas.T,dds0))/yexperr))
        for ide,de in enumerate(dds1):
            if de<0:dds1[ide]=0
            if de>200:dds1[ide]=200
        dds0=dds1
        print dds1'''
    return dsalscp['x']


def lsqmas(fullprint=False):
    u"""находим решение A(jac)x(param)-b(yexp)-min
    """
    startobrsolve = time.time()
    global shag
    dx0=np.zeros(len(mod_coef)+1) #отправная точка решения - все отклонения - нули
    dx0=np.ones(len(mod_coef)+1) #отправная точка решения - все отклонения - единицы
    dx0.fill(0.1)
    s_sum=[]
    shag=1
    y_from_model_mas=y_fr_model() # для ускорения считаем только 1 раз
    #ssnp=np.linalg.lstsq(mas.T,yexp1-y0m) #решение
    #ssfort=scipy.optimize.nnls(mas.T,yexp1-y0m)
    #sslsqscypy=scipy.optimize.leastsq(funk_fr_jac,mod_coef_delt)#np.array([inp_data.iloc[0][x] for x in mod_coef])
    sslsqscypy=scipy.optimize.leastsq(f4minimise_buf,dx0,epsfcn=0.01,maxfev =300, factor=0.1,full_output=1,args=(fullprint))#ftol=0.001,xtol=0.01,
    ss=sslsqscypy
    np.set_printoptions(suppress=True,precision=3,edgeitems=10)
    if fullprint:
        print mod_coef
        print 'solve dxnorm-',ss[0]
        print 'solve dx-',normolizeX_ob(ss[0][:-1])
        print 'solve x-',dx2x(normolizeX_ob(ss[0][:-1]))
        print 'Ppgadd -',ss[0][-1]
    xbounded,rrrrel_x=boundX(dx2x(normolizeX_ob(ss[0][:-1])),fullprint)
    finishobrsolve = time.time()
    if fullprint:
        print '____________________________________________________'
        print u"Скорость выполенения всего: ",(finishobrsolve - startobrsolve)/60.,u" минут"
        print u"число шагов",shag
        print '____________________________________________________'
    '''
    plt.plot(s_sum);plt.show()
    print 'var\texp\tsolve\tdiff\tdiff/deriv'
    for pp,vv in enumerate(yexpvar):
        print vv,'\t',
        print y_from_model_mas[pp],'\t',
        print y00[pp]+np.dot(mas.T,normolizeX_ob(ss[0][:-1]))[pp],'\t',
        print y_from_model_mas[pp]-y00[pp]-np.dot(mas.T,-x00+xbounded)[pp],'\t',
        print (y_from_model_mas[pp]-y00[pp]-np.dot(mas.T,-x00+xbounded)[pp])/arch_var_deviation[vv]
    print '___'
    stest=0
    for pp,vv in enumerate(yexpvar):
        #stest+=((y_from_model_mas[pp]-y00[pp]-np.dot(mas.T,-x00+xbounded)[pp])/arch_var_deviation[vv])**2
        stest1par=((Ppg_popravka(ss[0][-1]*4)[pp]-y00[pp]-np.dot(mas.T,-x00+xbounded)[pp])/arch_var_deviation[vv])**2
        stest+=stest1par
        print vv,'\t',
        print stest1par,'\t',
        print stest
    '''
    return ss[0]

def solve_throught_arch():
    u"""
    идем через архив и решаем задачу
    """
    global y0m
    jac,x0masfit=all_JAC()
    solvepointsm=range(0,5000,10)
    solvemass=[]
    ysolmas=[]
    startobrsolve = time.time()
    jdf=pd.DataFrame(jac)
    jdfn=(jdf.T/pd.Series(arch_var_deviation)).T
    JTnenorm=jac_2_matrix(jdf)  #NEnormir jac
    JT=jac_2_matrix(jdfn) #normir jac
    yexperr=np.array([arch_var_deviation[ee] for ee in yexpvar])
    JT_restr=restrict_JAC(JT)
    JTnenorm_restr=restrict_JAC(JTnenorm)
    yexperr_restr=restrict_yexperr(yexperr,JT_restr)
    y0_restr=restrict_y0(y0m,JT_restr)
    for spoi in solvepointsm:
        put_data_fileh5_model(spoi)
        print v['OG_T_pvd1'],v['OG_T_pvd2']
        yexp1=y_fr_model()
        yexp1_restr=restrict_yexp(yexp1,JT_restr)
        sol=newton_gauss(JT_restr,yexp1_restr,y0_restr,yexperr_restr,fullprint=False)
        solvemass.append(sol) #составляем массив решений задачи идя через архив
        ysol=func_by_delt(sol,JTnenorm_restr,y0_restr)
        ysolmas.append(ysol)
    finishobrsolve = time.time()
    print '____________________________________________________'
    print u"Скорость выполенения всего: ",(finishobrsolve - startobrsolve)/60.,u" минут"
    #print u"число шагов",shag
    print '____________________________________________________'

    #otchet:
    ysolnpm=np.array(ysolmas)
    #plt.plot(ysolnpm.T[0])
    for pp in solvepointsm:
        yexp_te=store['alldata'].iloc[range(0,1710,10)]['tgorp1_sr']
    ysol_df=pd.DataFrame(ysolnpm.T[0],store['alldata'].iloc[range(0,1710,10)].index,columns=[u'Решение обратной задачи'])
    yexp_te.name=u'Эксперимент'
    ysol_df.plot()
    #ysol_df.iloc[range(0,ysol_df.shape[0],10)].plot(yerr=3,style='+')
    yexp_te.plot(legend=True,style='--')
    #yexp_te.iloc[range(5,yexp_te.shape[0],10)].plot(style='r+',yerr=1)
    #plt.style='red'
    plt.xlabel(u"Время в архиве",fontsize=16)
    plt.ylabel(yexpvar_lable[u"Tgor1"],fontsize=16)
    plt.show()
    (ysol_df.icol(0)-yexp_te).plot()
    plt.hist(ysol_df.icol(0)-yexp_te,bins=20)
    plt.xlabel(ur'$\Delta$'+yexpvar_lable[u"Tgor1"],fontsize=16)
    plt.show()
    print (ysol_df.icol(0)-yexp_te).mean(),(ysol_df.icol(0)-yexp_te).std() #sistem pogreshnost
    #N1k
    for pp in solvepointsm:
        yexp_te=store['alldata'].iloc[range(0,1710,10)]['N1k']
    ysol_df=pd.DataFrame(ysolnpm.T[14],store['alldata'].iloc[range(0,1710,10)].index,columns=[u'Решение обратной задачи'])
    yexp_te.name=u'Эксперимент'
    ysol_df.plot()
    #ysol_df.iloc[range(0,ysol_df.shape[0],10)].plot(yerr=3,style='+')
    yexp_te.plot(legend=True,style='--')
    #yexp_te.iloc[range(5,yexp_te.shape[0],10)].plot(style='r+',yerr=1)
    #plt.style='red'
    plt.xlabel(u"Время в архиве",fontsize=16)
    plt.ylabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    plt.xlabel(u"Время в архиве",fontsize=16)
    plt.ylabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    (ysol_df.icol(0)-yexp_te).plot()
    plt.hist(ysol_df.icol(0)-yexp_te,bins=20)
    plt.xlabel(ur'$\Delta$'+yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    print (ysol_df.icol(0)-yexp_te).mean(),(ysol_df.icol(0)-yexp_te).std() #sistem pogreshnost


def search_derive_solver():
    u"""
    Ищем погрешность переменных (мощности в тч) после решения обр задачи
    1. задаем погрешность входных данных
    2. решаем по массиву с этой погрешностью
    3.
    4. профит!
    """
    startobrsolvef = time.time()
    #создаем массив значений вокруг эксперимент точки на которую натягиваем модель, массив со стандартным отклонением по погрешности
    solvedfall={}
    ysolvedfall={}
    #yexp1=Ppg_popravka(0)
    #yexperr=np.array([arch_var_deviation[ee] for ee in yexpvar])
    for iy,y in enumerate(yexpvar):
        print iy,y,y_from_model_mas[iy],arch_var_deviation[y]
        print arch_var_deviation[y]*100./y_from_model_mas[iy]
        tempmas=np.random.normal(y_from_model_mas[iy],arch_var_deviation[y],400) #набираем случайных значений
        ##tempmas=np.random.normal(v.OG_T_pvd1,arch_var_deviation['tpvd1'],10)
        solvemass=[]
        for tpar in tempmas:
            y_from_model_mas=y_fr_model() #берем решение из модели
            y_from_model_mas[iy]=tpar #кладем туда возмущенные параметры
            solvemass.append(newton_gauss(fullprint=False))
        solvedf=pd.DataFrame(solvemass,tempmas) #массив решений dx
        solvedf.index.name=y
        ysolvemass=[]
        for sspoi in solvemass:
            dxsol=sspoi
            #xsol=dx2x(normolizeX_ob(dxsol[:-1]))
            #xsolb,rrrrrr=boundX(xsol,fullprint=False)
            #ysol=y00+np.dot(mas.T,xsolb-x00)
            ysol=func_by_delt(dxsol)
            ysolvemass.append(ysol)
        yexpvarpsev=list(yexpvar)
        yexpvarpsev.extend(['psev1_1','psev1_2','psev1_3','psev2_1','psev2_2','psev2_3','psev3_1','psev3_2','psev3_3'])
        ysolvedf=pd.DataFrame(ysolvemass,tempmas,yexpvarpsev) #массив парам модели после решений
        ysolvedf.index.name=y
        solvedfall[y]=solvedf
        ysolvedfall[y]=ysolvedf
        print ysolvedfall[y].std()*100./ysolvedfall[y].mean()
    #storeofds = pd.HDFStore(dirofdis+'OG_T_pvd1mas10k.py')
    #storeofds['ysolvedf']=ysolvedf
    #storeofds['solvedf']=solvedf
    #storeofds.close()
    finishobrsolvef = time.time()
    print '____________________________________________________'
    print u"Скорость выполенения всего: ",(finishobrsolvef - startobrsolvef)/60.,u" минут"
    #print u"число шагов",shag
    print '____________________________________________________'
    print 'std=',ysolvedfall['Tgor1']['Tgor1'].std()
    print 'std=',ysolvedfall['Tgor1']['Tgor1'].mean()
    print 'std=',ysolvedfall['Tgor1']['Tgor1'].std()*100/ysolvedfall['Tgor1']['Tgor1'].mean()
    plt.hist(ysolvedfall['Tgor1']['Tgor1'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"Tgor1"],fontsize=16)
    plt.show()
    plt.hist(ysolvedfall['Tgor1']['Tgor1'].index,bins=40)
    plt.xlabel(yexpvar_lable[u"Tgor1"],fontsize=16)
    plt.show()
    print 'std=',ysolvedfall['Tgor1']['N1k'].std()
    print 'std=',ysolvedfall['Tgor1']['N1k'].mean()
    print 'std=',ysolvedfall['Tgor1']['N1k'].std()*100/ysolvedfall['Tgor1']['N1k'].mean()
    plt.hist(ysolvedfall['Tgor1']['N1k'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    plt.hist(ysolvedfall['N1k']['N1k'].index,bins=40)
    plt.xlabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    plt.hist(ysolvedfall['N1k']['N1k'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    plt.hist(ysolvedfall['N1k']['Tgor1'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"Tgor1"],fontsize=16)
    plt.show()

    print 'std=',ysolvedfall['Pgpk']['Pgpk'].std()
    print 'std=',ysolvedfall['Pgpk']['Pgpk'].mean()
    print 'std=',ysolvedfall['Pgpk']['Pgpk'].std()*100/ysolvedfall['Pgpk']['Pgpk'].mean()
    plt.hist(ysolvedfall['Pgpk']['Pgpk'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"Pgpk"],fontsize=16)
    plt.show()
    plt.hist(ysolvedfall['Pgpk']['Pgpk'].index,bins=40)
    plt.xlabel(yexpvar_lable[u"Pgpk"],fontsize=16)
    plt.show()
    print 'std=',ysolvedfall['Pgpk']['N1k'].std()
    print 'std=',ysolvedfall['Pgpk']['N1k'].mean()
    print 'std=',ysolvedfall['Pgpk']['N1k'].std()*100/ysolvedfall['Pgpk']['N1k'].mean()
    plt.hist(ysolvedfall['Pgpk']['N1k'].values,bins=40)
    plt.xlabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()


    '''
    for iy,y in enumerate(yexpvar):
        print iy,y,arch_var_deviation[y]*100./y_from_model_mas[iy]
        print ysolvedfall[y].std()*100./ysolvedfall[y].mean()
    '''
    #шумим не по одной переменной, а всеми разом
    startobrsolvef = time.time()
    u"""создаем массив значений вокруг эксперимент точки на которую натягиваем модель, массив со стандартным отклонением по погрешности"""
    solvedfall={}
    ysolvedfall={}
    y_from_model_mas=y_fr_model()
    y_from_model_mas_o=y_fr_model()
    y_new=[]
    for iy,y in enumerate(yexpvar):
        #print iy,y,y_from_model_mas[iy],arch_var_deviation[y]
        #print arch_var_deviation[y]*100./y_from_model_mas[iy]
        tempmas=np.random.normal(y_from_model_mas[iy],arch_var_deviation[y],10000)
        ##tempmas=np.random.normal(v.OG_T_pvd1,arch_var_deviation['tpvd1'],10)
        #solvemass=[]
        #for tpar in tempmas:
        #y_from_model_mas=y_fr_model()
        y_new.append(tempmas)
    y_new=np.array(y_new)
    solvemass=[]
    for i in range(y_new.shape[1]):
        y_from_model_mas=y_new[:,i]
        solvemass.append(newton_gauss(fullprint=False))
    solvedf=pd.DataFrame(solvemass) #массив решений dx
    ysolvemass=[]
    for sspoi in solvemass:
        dxsol=sspoi
        #xsol=dx2x(normolizeX_ob(dxsol[:-1]))
        #xsolb,rrrrrr=boundX(xsol,fullprint=False)
        #ysol=y00+np.dot(mas.T,xsolb-x00)
        ysol=func_by_delt(dxsol)
        ysolvemass.append(ysol)
        ysolvemass.append(ysol)
    yexpvarpsev=list(yexpvar)
    yexpvarpsev.extend(['psev1_1','psev1_2','psev1_3','psev2_1','psev2_2','psev2_3','psev3_1','psev3_2','psev3_3'])
    ysolvedf=pd.DataFrame(ysolvemass,columns=yexpvarpsev) #массив парам модели после решений
    print ysolvedf.std()*100./ysolvedf.mean()
    finishobrsolvef = time.time()
    print '____________________________________________________'
    print u"Скорость выполенения всего: ",(finishobrsolvef - startobrsolvef)/60.,u" минут"
    #print u"число шагов",shag
    print '____________________________________________________'
    ysolvedf['N1k'].hist(bins=40)
    plt.xlabel(yexpvar_lable[u"N1k"],fontsize=16)
    plt.show()
    for iy,y in enumerate(yexpvar):
        print '{0:2d} {1:7s} {2:7.3f} {3:7.3f}'.format(iy, y, ysolvedf[y].std(),arch_var_deviation[y]) # {1} {5.3f} {45.3f}
    #solvedfall[y]=solvedf
    #ysolvedfall[y]=ysolvedf
        #solvemass.append(lsqmas(fullprint=False))

def sumsq_vliyan():
    u"""
    составление таблицы влияния на сумму квадратов от изменения параметров модели
    """
    dx0=np.zeros(len(mod_coef)+1)
    sumvlel=f4minimise_buf(dx0,fullprint=False)
    sumvl=0
    for el in sumvlel:
        sumvl+=el*el
    sumbeg=sumvl
    for dxel in range(len(dx0)):
        try:
            print mod_coef[dxel],
        except IndexError:
            print 'popravka ppg',
        dx0=np.zeros(len(mod_coef)+1)
        dx0[dxel]=-1.
        sumvlel=f4minimise_buf(dx0,fullprint=False)
        sumvl=0
        for el in sumvlel:
            sumvl+=el*el
        sumend=sumvl
        sumder=sumend-sumbeg
        print '\t',sumder

def oform_diss_exp_data():
    u"""
    оформление дисс, эсперимент данные
    """
    print store['alldata']['N1k'].index #date arch
    store['alldata']['N1k'].plot()
    plt.ylabel(yexpvar_lable[u"N1k"]+ur'$, МВт$',fontsize=16)
    plt.xlabel(u"Время в архиве",fontsize=16)
    plt.show()




def main():
    pass

if __name__ == '__main__':
    main()

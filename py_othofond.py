#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import re

inpfile="G:/py_fond/ex1f.xlsx"

rb=xlrd.open_workbook(inpfile)
sheet=rb.sheet_by_index(0)

for i,el in enumerate(sheet.row_values(11)):
    print i,el #print head of doc

for irow in range(11+1,sheet.nrows-1):
    for iel in sheet.row_values(irow):
        print iel #printing all doc

def balance():
    u"""
    Считаем дебет, кредет, баланс
    """
    debet=0
    for irow in range(11+1,sheet.nrows-1):
        """summ of DEBET (rashod)
        """
        if sheet.row_values(irow)[7]!='':
            print irow, sheet.row_values(irow)[7]
            debet+=sheet.row_values(irow)[7]
    print 'itogo debet = ',debet

    credet=0
    for irow in range(11+1,sheet.nrows-1):
        """summ of CREDET (prihod)
        """
        if sheet.row_values(irow)[8]!='':
            print irow, sheet.row_values(irow)[8]
            credet+=sheet.row_values(irow)[8]
    print 'itogo credet = ',credet

    balance=credet-debet
    print balance


def anonim_name(full_name):
    u"""
    анонимизируем имя из банка
    """
    fn=full_name.split()
    return fn[1]+' '+fn[2]+' '+fn[0][0]+'.'


def names_donors():
    u"""
    обрабатываем имена жертвователей
    """
    for irow in range(11+1,sheet.nrows-1):
        if sheet.row_values(irow)[8]!='':
            if len(sheet.row_values(irow)[4].split('//'))==5:
            #print irow, sheet.row_values(irow)[4],len(sheet.row_values(irow)[4].split('//'))
                print irow, anonim_name(sheet.row_values(irow)[4].split('//')[1]),'\t',sheet.row_values(irow)[8]
            elif len(sheet.row_values(irow)[4].split('//'))==3:
                print irow, anonim_name(sheet.row_values(irow)[4].split('//')[0]),'\t',sheet.row_values(irow)[8]
            else:
                print irow, sheet.row_values(irow)[4],len(sheet.row_values(irow)[4].split('//')),sheet.row_values(irow)[8]




'''pattern=re.compile(u"ИЛ1ЬЯ|АЛЕ",re.IGNORECASE)
q=pattern.search(sheet.row_values(39)[4].split('//')[1])
if q:
    print q.group()
else:
    print "Nooo!"
'''

def main():
    pass

if __name__ == '__main__':
    main()

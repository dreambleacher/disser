#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xlrd
import re

inpfile="D:/ex1f.xlsx"
outfile="D:/outfond.txt"

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
    outfn=fn[1]+' '+fn[2]+' '+fn[0][0]+'.'
    return outfn


def names_donors():
    u"""
    обрабатываем имена жертвователей
    """
    f=open(outfile,'w')
    f.write(r'<ul>'+'\n\t'+r'<li style="list-style-type: none;">'+'\n\t\t'+r'<table border="1" style="width: 100%;">'+'\n\t\t\t'+r'<tbody>'+'\n')
    for irow in range(sheet.nrows-2,11+1,-1): #(11+1,sheet.nrows-1)
        if sheet.row_values(irow)[8]!='':
            if len(sheet.row_values(irow)[4].split('//'))==5:
            #print irow, sheet.row_values(irow)[4],len(sheet.row_values(irow)[4].split('//'))
                print '{0}\t{1}\t{2}\t{3}'.format(irow,sheet.row_values(irow)[1], anonim_name(sheet.row_values(irow)[4].split('//')[1]).encode('cp1251'),sheet.row_values(irow)[8])
                f.write('\t\t\t'+r'<tr>'+'\n')
                f.write('\t\t\t<td>{1}</td>\t<td>{2}</td>\t<td>{3}</td>\n'.format(irow,sheet.row_values(irow)[1], anonim_name(sheet.row_values(irow)[4].split('//')[1]).encode('cp1251'),sheet.row_values(irow)[8]))
                f.write('\t\t\t'+r'</tr>'+'\n')
            elif len(sheet.row_values(irow)[4].split('//'))==3:
                print irow,sheet.row_values(irow)[1], anonim_name(sheet.row_values(irow)[4].split('//')[0]).encode('cp1251'),'\t',sheet.row_values(irow)[8]
                f.write('\t\t\t'+r'<tr>'+'\n')
                f.write('\t\t\t<td>{1}</td>\t<td>{2}</td>\t<td>{3}</td>\n'.format(irow,sheet.row_values(irow)[1], anonim_name(sheet.row_values(irow)[4].split('//')[0]).encode('cp1251'),sheet.row_values(irow)[8]))
                f.write('\t\t\t'+r'</tr>'+'\n')
            else:
                print irow,sheet.row_values(irow)[1], sheet.row_values(irow)[4],len(sheet.row_values(irow)[4].split('//')),sheet.row_values(irow)[8]
                f.write('\t\t\t'+r'<tr>'+'\n')
                f.write('\t\t\t<td>{1}</td>\t<td>{2}</td>\t<td>{4}</td>\n'.format(irow,sheet.row_values(irow)[1], sheet.row_values(irow)[4].encode('cp1251'),len(sheet.row_values(irow)[4].split('//')),sheet.row_values(irow)[8]))
                f.write('\t\t\t'+r'</tr>'+'\n')
    f.write('\t\t'+r'</tbody>'+'\n\t'+r'</table></li>'+'\n'+r'</ul>')
    f.close()





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

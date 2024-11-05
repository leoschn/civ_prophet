from os.path import split

import pandas as pd
from unidecode import unidecode
from scraping import retrieve_messages
from constants.constant import DRAFT_TEMPLATE, MAP_DICT, CIV_DICT, ROLE_ID


def extract_from_string(s, verbose=False):
    data = DRAFT_TEMPLATE
    splited_s = unidecode(s).lower()
    #replace team
    for k in ROLE_ID.keys():
        splited_s = splited_s.replace(k,ROLE_ID[k])
    splited_s = splited_s.split('\n')
    splited_s = [i for i in splited_s if i != '']

    dec = 0

    #check if message is a report and extract winner
    if not 'vs' in splited_s[0].replace('team',''):
        dec = 1

    #ban sur 2 lignes
    if '/' in splited_s[4+dec].strip() and '/' in splited_s[5+dec].strip() :
        splited_s = splited_s[:4]+[splited_s[4]+splited_s[5]]+splited_s[6:]

    if 'vs' in splited_s[0+dec]:
        if splited_s[5+dec].strip() in splited_s[1+dec]:
            data['Winner']=1
        elif splited_s[10+dec].strip() in splited_s[1+dec]:
            data['Winner']=2
        else :
            if verbose:
                print('No winner found', splited_s[5+dec].strip(),splited_s[1+dec], splited_s[10+dec].strip(), splited_s[1+dec])
                print(s)
            raise 'No winner found'


        #extract map

        data['Map played']=MAP_DICT[splited_s[2+dec].strip()]

        #extract map bans

        if 'map' in splited_s[3+dec]:

            line = splited_s[3+dec].split(':')
            bans = line[1].split('/')
            for i in range(6) :
                try :
                    data['Map ban{0}'.format(i+1)]=MAP_DICT[bans[i].strip()]
                except :
                    data['Map ban{0}'.format(i+1)]=0

        #extract leader bans


        if 'leader' in splited_s[4+dec]:

            line = splited_s[4+dec].split(':')
            bans = line[1].split('/')
            for i in range(14) :
                try :
                    data['Ban{0}'.format(i+1)]=CIV_DICT[bans[i].strip()]
                except :
                    data['Ban{0}'.format(i+1)]=0


            #extract leaders picks

        if splited_s[5+dec].strip() in splited_s[1+dec]:
            try :
                data['PickW1']=CIV_DICT[' '.join(splited_s[6+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW1'] = 0
            try:
                data['PickW2'] = CIV_DICT[' '.join(splited_s[7+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW2'] = 0
            try :
                data['PickW3']=CIV_DICT[' '.join(splited_s[8+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW3'] = 0
            try :
                data['PickW4']=CIV_DICT[' '.join(splited_s[9+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW4'] = 0
            try:
                data['PickL1'] = CIV_DICT[' '.join(splited_s[11+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL1'] = 0
            try:
                data['PickL2'] = CIV_DICT[' '.join(splited_s[12+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL2'] = 0
            try:
                data['PickL3'] = CIV_DICT[' '.join(splited_s[13+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL3'] = 0
            try:
                data['PickL4'] = CIV_DICT[' '.join(splited_s[14+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL4'] = 0
        elif splited_s[10+dec].strip() in splited_s[1+dec]:
            try:
                data['PickL1'] = CIV_DICT[' '.join(splited_s[6+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL1'] = 0
            try:
                data['PickL2'] = CIV_DICT[' '.join(splited_s[7+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL2'] = 0
            try:
                data['PickL3'] = CIV_DICT[' '.join(splited_s[8+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL3'] = 0
            try:
                data['PickL4'] = CIV_DICT[' '.join(splited_s[9+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickL4'] = 0
            try :
                data['PickW1']=CIV_DICT[' '.join(splited_s[11+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW1'] = 0
            try:
                data['PickW2'] = CIV_DICT[' '.join(splited_s[12+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW2'] = 0
            try :
                data['PickW3']=CIV_DICT[' '.join(splited_s[13+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW3'] = 0
            try :
                data['PickW4']=CIV_DICT[' '.join(splited_s[14+dec].split(' ')[1:]).strip().split('<')[0].strip()]
            except:
                data['PickW4'] = 0
        else:
            if verbose:
                print('Team not consistant with front')
                print(s)
            raise 'Team not consistant with front'
    else:
        if verbose:
            print('Matching failed')
            print(s)
        raise 'Matching failed'

    return data

def extract_from_serie(s, verbose=False):
    l=[]
    for i in range(len(s)):
        try :
            data = extract_from_string(s[0][i], verbose)
        except :
            pass
        else:
            l.append(data.copy())
    return pd.DataFrame(l)


if __name__ == '__main__' :
    test = retrieve_messages('952006558454272040')
    l = extract_from_serie(test)
    # l= extract_from_string(test[0][86])
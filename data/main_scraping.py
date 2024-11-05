from preprocessing import extract_from_serie
from scraping import retrieve_messages
import pandas as pd

""" channel ids : 
951171373940752424 D1
952006558454272040 D2
951171316512346162 D3
1061362182828326922 D4
"""


if __name__ == '__main__' :
    msg1 = retrieve_messages('951171373940752424',100, '## S14\nhttps://challonge.com/fr/SquadronCivFrS14D1')
    df1 = extract_from_serie(msg1)
    df1['Division']  = 1
    msg2 = retrieve_messages('952006558454272040', 100, '## S14\nhttps://challonge.com/fr/SquadronCivFrS14D2')
    df2 = extract_from_serie(msg2)
    df2['Division'] = 2
    msg3 = retrieve_messages('951171316512346162', 100, '## S14\nhttps://challonge.com/fr/SquadronCivFrS14D3')
    df3 = extract_from_serie(msg3)
    df3['Division'] = 3
    msg4 = retrieve_messages('1061362182828326922', 100, '## S14\nhttps://challonge.com/fr/SquadronCivFrS14D4')
    df4 = extract_from_serie(msg4)
    df4['Division'] = 4
    df = pd.concat([df1,df2,df3,df4], axis=0)
    df.to_csv('data_S14_07_10.csv')
import requests
import json
import pandas as pd
from sympy.utilities.iterables import bracelets


def retrieve_messages(channelid, max_lim, stop_value):
    list_msg=[]
    num=0
    max_num = max_lim
    limit = 1
    headers = {
        'authorization': 'PERSONAL TOKEN'
    }
    last_message_id = None
    while num<max_num:
        query_parameters = f'limit={limit}'
        if last_message_id is not None:
            query_parameters += f'&before={last_message_id}'
        r = requests.get(
            'https://discord.com/api/v9/channels/{0}/messages?{1}'.format(channelid,query_parameters),headers=headers
            )
        jsonn = json.loads(r.text)

        if len(jsonn) == 0:
            break

        for value in jsonn:
            last_message_id = value['id']
            num = num + 1


        for value in jsonn:
            if stop_value in value['content'] :
                num = max_num
                break
            list_msg.append(value['content'])

    return pd.DataFrame(list_msg)



if __name__ == '__main__' :
    test = retrieve_messages('1061362182828326922',100,'## S14\nhttps://challonge.com/fr/SquadronCivFrS14D4')

import json
import pandas as pd
import numpy as np
import datetime
from collections import Counter

##TODAY's Date
TODAY = datetime.datetime(year=2017, month=1, day=1)

##Fetching the data from the JSON training file
def importDataFromJSON(filepath):
    ls = []
    with open(filepath) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls

def division_handler(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0

def before_today(date_obj):
    return date_obj < TODAY

##Creating the dataset dictionary
class ConvertData:
    def __init__(self, filepath):
        self.target = []
        self.dict_lst = []
        self.id = []
        self.list = importDataFromJSON(filepath)
        return


    def getAge(self, bday):
        return int(((TODAY - datetime.datetime.strptime(bday, '%Y-%m-%d')).days) / 365)

    def checkMale(self, ismale):
        if ismale:
            return 1
        else:
            return 0


    def resourceCal(self, resources):
        cnt_1 = Counter()
        cnt_2 = Counter()
        for resource in resources:
            if before_today(datetime.datetime.strptime(resource, '%Y-%m-%d')):
                for item in resources[resource]:
                    if item is not None:
                        cnt_1[item.split('_')[0]] += 1
                        cnt_2[item] += 1
        return (cnt_1, cnt_2)


    def resourceLen(self, resources):
        recency = -1
        cnt_2 = Counter()
        res_days = 0
        min_date_str = '2020-01-01'
        max_date_str = '2000-01-01'
        min_date_obj = datetime.datetime.strptime(min_date_str, '%Y-%m-%d')
        max_date_obj = datetime.datetime.strptime(max_date_str, '%Y-%m-%d')
        valid = 0
        for resource in resources:
            if before_today(datetime.datetime.strptime(resource, '%Y-%m-%d')):
                try:
                    valid += 1
                    res_date_obj = datetime.datetime.strptime(resource, '%Y-%m-%d')
                    min_date_obj = min(res_date_obj, min_date_obj)
                    max_date_obj = max(res_date_obj, max_date_obj)
                    cnt_2['hospital_visits'] += 1
                except:
                    print("An exception occurred")
        if valid > 0:
            res_days = (max_date_obj - min_date_obj).days
            recency = (TODAY - max_date_obj).days
            for item in resources[max_date_obj.strftime('%Y-%m-%d')]:
                cnt_2[('last_'+item)] += 1
            #for item in resources[min_date_obj.strftime('%Y-%m-%d')]:
                #cnt_2[('first_'+item)] += 1
        cnt_2['res_days'] = res_days
        cnt_2['res_recency'] = recency
        return cnt_2

    def observaionCal(self, observations):
        high = normal = low = abnormal = missing = 0
        sum_high = sum_normal = sum_low = sum_abnormal = 0
        cnt = Counter()
        for observation in observations:
            if before_today(datetime.datetime.strptime(observation, '%Y-%m-%d')):
                for dictionary in observations[observation]:
                    cnt[dictionary['code']] += 1
                    if dictionary['interpretation'] is not None:
                        cnt[(dictionary['code']+'_'+dictionary['interpretation'])] += 1
                        cnt[dictionary['interpretation']] += 1
        return cnt

    def observaionLen(self, observations):
        obs_days = 0
        min_date_str = '2020-01-01'
        max_date_str = '2000-01-01'
        min_date_obj = datetime.datetime.strptime(min_date_str, '%Y-%m-%d')
        max_date_obj = datetime.datetime.strptime(max_date_str, '%Y-%m-%d')
        valid = 0
        recency = -1
        cnt = Counter()

        for observation in observations:
            if before_today(datetime.datetime.strptime(observation, '%Y-%m-%d')):
                try:
                    valid += 1
                    obs_date_obj = datetime.datetime.strptime(observation, '%Y-%m-%d')
                    min_date_obj = min(obs_date_obj, min_date_obj)
                    max_date_obj = max(obs_date_obj, max_date_obj)
                    cnt['lab_visits'] += 1
                except:
                    print("An exception occurred")
        if valid > 0:
            obs_days = (max_date_obj - min_date_obj).days
            recency = (TODAY - max_date_obj).days
            for dictionary in observations[max_date_obj.strftime('%Y-%m-%d')]:
                cnt[('last_'+dictionary['code'])] += 1
                if dictionary['value'] is not None:
                    cnt[('last_'+dictionary['code']+'_val')] = float(dictionary['value'])
            #for dictionary in observations[min_date_obj.strftime('%Y-%m-%d')]:
                #cnt[('first_'+dictionary['code'])] += 1
                #if dictionary['value'] is not None:
                    #cnt[('first_'+dictionary['code']+'_val')] = float(dictionary['value'])

            cnt['obs_recency'] = recency
            cnt['obs_days'] = obs_days
        return cnt

    def targetValid(self, tag_dm2):
        if(tag_dm2 == '') :
            valid, target = True, 0
        elif datetime.datetime.strptime('2017-01-01', '%Y-%m-%d') <= datetime.datetime.strptime(tag_dm2, '%Y-%m-%d') < datetime.datetime.strptime('2018-01-01', '%Y-%m-%d'):
            valid, target = True, 1
        else:
            valid, target = False, np.nan
        return (valid, target)

    def createTrainDictLst(self, data_set_name=None):
        for record in self.list:
            (valid, target) = self.targetValid(record['tag_dm2'])
            if valid:

                age = self.getAge(record['bday'])
                ismale = self.checkMale(record['is_male'])

                age_counter = Counter({'Age':age})
                gender_counter = Counter({'IsMale':ismale})
                (resourceCal_counter_1, resourceCal_counter_2) = self.resourceCal(record['resources'])
                resourceLen_counter = self.resourceLen(record['resources'])
                observaionCal_counter = self.observaionCal(record['observations'])
                observaionLen_counter = self.observaionLen(record['observations'])
                aggre_dict = dict(age_counter + gender_counter + resourceLen_counter + resourceCal_counter_1 + resourceCal_counter_2 + observaionCal_counter + observaionLen_counter)
                self.dict_lst.append(aggre_dict)
                self.target.append(target)
        with open(data_set_name, 'w') as fp:
            json.dump({'features':self.dict_lst, 'target':self.target}, fp)
            print('create Train Data set')



    def createTestDictLst(self, data_set_name=None):
        for record in self.list:
            if True:
                age = self.getAge(record['bday'])
                ismale = self.checkMale(record['is_male'])

                age_counter = Counter({'Age':age})
                gender_counter = Counter({'IsMale':ismale})
                (resourceCal_counter_1, resourceCal_counter_2) = self.resourceCal(record['resources'])
                resourceLen_counter = self.resourceLen(record['resources'])
                observaionCal_counter = self.observaionCal(record['observations'])
                observaionLen_counter = self.observaionLen(record['observations'])
                aggre_dict = dict(age_counter + gender_counter + resourceLen_counter + resourceCal_counter_1 + resourceCal_counter_2 + observaionCal_counter + observaionLen_counter)
                self.dict_lst.append(aggre_dict)
                self.id.append(record['patient_id'])
        with open(data_set_name, 'w') as fp:
            json.dump({'features':self.dict_lst, 'patient_id':self.id}, fp)
            print('create Test Data set')

    def createDictLst(self, is_train=True, data_set_name=None):
        if is_train:
            self.createTrainDictLst(data_set_name)
        else:
            self.createTestDictLst(data_set_name)


#Creating the training dataset
training_instance = ConvertData('train.txt')
training_instance.createDictLst(is_train= True, data_set_name= 'processed_train_dataset.json')
print('Train json file generated and asved at same folder called processed_train_dataset.json')


#Creating the test Dataset
test_instance = ConvertData('test.txt')
test_instance.createDictLst(is_train=False, data_set_name='processed_test_dataset.json')
print('Test json file generated and asved at same folder called processed_test_dataset.json')

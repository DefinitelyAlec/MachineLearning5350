# must support information gain, majority error, gini index
# allow user to set max tree depth
import math
from sre_constants import SUCCESS

# Node class for decision tree
class Node:
    def __init__(self) -> None:
        self.children = {}
        self.attr = None
        self.index = -1
    
    def append(self, value, node):
        self.children[value] = node
    
    def walkNodes(self, path):
        if self.index == -1:
            if path[len(path)-1] != self.attr:
                return False
            else: 
                return True
        return self.children.get(
            path[self.index]).walkNodes(path)

#----------Helper methods below----------

# Calculate the occurance of a certain value given a list of values
def probability(list, toCalc):
    occurance = 0
    if len(list) == 0:
        # avoid dividing by 0
        return 0

    for l in list:
        if l[len(l)-1] == toCalc:
            occurance += 1
    return occurance / len(list)

# Calculate the entropy given a list and its sublist of labels
def Entropy(list, labels):
    total = 0
    for l in labels:
        p = probability(list, l)
        if p == 0:
            # dont do anything, trying to take log2 of 0 is bad
            pass
        else:
            total += p*math.log(p, 2)
    return -total

# Calculate the MajorityError given a list and its sublist of labels
def MajorityError(list, labels):
    maxCountSoFar = 1
    for l in labels:
        count = 0
        for row in list:
            if row[len(row)-1] == l:
                count += 1
        if count > maxCountSoFar:
            maxCountSoFar = count
    return 1 - (maxCountSoFar/len(list))

# Calculate the Gini Index given a list and its sublist of labels
def GiniIndex(list, labels):
    total = 0
    for l in labels:
        p = probability(list, l)
        total += math.pow(p, 2)
    return 1 - total

# Calculate the information gain
def InformationGain(list, KV, labels, heuristic):
    index, vals = KV
    total = 0
    for v in vals:
        sublist = []
        for l in list:
            if l[index] == v:
                sublist.append(l)
        total += (len(sublist)/len(list)) * (heuristic(sublist, labels))
    return heuristic(list, labels)-total

#----------Main algorithm----------

# Implementation of ID3
def ID3(list, attributes, label, max_depth=-1, heuristic="GI"):
    # Check if every example is the same, if so no need for a split, return the root
    root = Node()
    first = list[0][len(list[0])-1]
    equalityFlag = True
    for l in list:
        if l[len(l)-1] != first:
            equalityFlag = False
            break
    if equalityFlag:
        root.attr = first
        return root
    
    # Calculate the best Information Gain among the labels

    func = None
    if heuristic == "GI":
        func = GiniIndex
    elif heuristic == "ME":
        func = MajorityError
    elif heuristic == "Entropy":
        func = Entropy
    else:
        func = None
    maxIG = 0
    KV = None
    for k in attributes.keys():
        i, v = attributes.get(k)
        IG = InformationGain(list, (i,v), label, func)
        if IG > maxIG:
            maxIG = IG
            KV = k
    
    # Check if only a single layer in tree, or no information gain was obtained
    if max_depth == 0 or KV == None:
        maxOccuranceVal = None
        maxOccurance = 0
        for v in label:
            count = 0
            for l in list:
                if l[len(v)-1] == v:
                    count += 1
            if count > maxOccurance:
                maxOccurance = count
                maxOccuranceVal = v
        root.attr = maxOccuranceVal
        return root

    # No issues so far, split on the best Information Gain from earlier
    i, vals = attributes.get(KV)
    root.attr = KV
    root.index = i
    for v in vals:
        sublist = []
        for l in list:
            if l[i] == v:
                sublist.append(l)

        # start finding leaf nodes
        if len(sublist) == 0:
            maxOccuranceVal2 = None
            maxOccurance2 = 0
            for v2 in label:
                count2 = 0
                for l in list:
                    if l[len(l)-1] == v2:
                        count2 += 1
                if count2 > maxOccurance2:
                    maxOccuranceVal2 = v2
                    maxOccurance2 = count2
            leaf = Node()
            leaf.attr = maxOccuranceVal2
            root.append(v, leaf)
        else:
            # need to split again
            tempAttr = attributes.copy()
            del tempAttr[KV]
            root.append(v, ID3(sublist, tempAttr, label, max_depth-1))
    return root

#----------Car Data----------

# build the car training data set
car_train_data = []
car_labels = ['unacc', 'acc', 'good', 'vgood']
car_attrs = {'buying': (0, ['vhigh', 'high', 'med', 'low']),
            'maint': (1, ['vhigh', 'high', 'med', 'low']),
            'doors': (2, ['2', '3', '4', '5more']),
            'persons': (3, ['2', '4', 'more']),
            'lugboot': (4, ['small', 'med', 'big']),
            'safety': (5, ['low', 'med', 'high'])}

with open('./DecisionTree/Car/train.csv', 'r') as car_train:
    for line in car_train:
        terms = line.strip().split(',')
        # process one training example
        car_train_data.append(terms)

# build the car test data set
car_test_data = []

with open('./DecisionTree/Car/test.csv', 'r') as car_test:
    for line in car_test:
        terms = line.strip().split(',')
        # process one training example
        car_test_data.append(terms)

# test the car data
func = "GI"
print(f'Information Gain type: {func}')
for a in range(len(car_attrs)):
    root = ID3(car_train_data, car_attrs, car_labels, a, heuristic=func)
    succ = 0
    fail = 0
    for row in car_test_data:
        if root.walkNodes(row):
            succ += 1
        else:
            fail += 1
    print(f'Max Depth: {a+1}, Error: {fail/(succ+fail)}')

#----------Bank Data----------

# build the bank training data set
bank_train_data = []
bank_labels = ['yes','no']
bank_attrs = {'age': (0, ['1', '0']),
                'job': (1, ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 
                'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 
                'technician', 'services']),
                'marital': (2, ['married','divorced','single']),
                'education': (3, ['unknown','secondary','primary','tertiary']),
                'default': (4, ['yes','no']),
                'balance': (5, ['1','0']),
                'housing': (6, ['yes','no']),
                'loan': (7, ['yes','no']),
                'contact': (8, ['unknown','telephone','cellular']),
                'day': (9, ['1', '0']),
                'month': (10, ['jan', 'feb', 'mar','apr','may','jun','jul','aug','sep','oct','nov','dec']),
                'duration': (11, ['1','0']),
                'campaign': (12, ['1','0']),
                'pdays': (13, ['1','0']),
                'previous': (14, ['1','0']),
                'poutcome': (15, ['unknown','other','failure','success'])}

ages = []
balances = []
days = []
durations = []
campaigns = []
pdays = []
previous = []

with open('./DecisionTree/Bank/train.csv', 'r') as bank_train:
    for line in bank_train:
        terms = line.strip().split(',')
        ages.append(terms[0])
        balances.append(terms[5])
        days.append(terms[9])
        durations.append(terms[11])
        campaigns.append(terms[12])
        pdays.append(terms[13])
        previous.append(terms[14])
    
    ages.sort()
    balances.sort()
    days.sort()
    durations.sort()
    campaigns.sort()
    pdays.sort()
    previous.sort()
    bank_train.seek(0)
    for line in bank_train:
        tempList = []
        terms = line.strip().split(',')
        if terms[0] > ages[int(len(ages)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        if terms[1] == 'unknown':
            tempList.append('blue-collar')
        else:
            tempList.append(terms[1])
        tempList.append(terms[2])
        if terms[3] == 'unknown':
            tempList.append('secondary')
        else:
            tempList.append(terms[3])
        tempList.append(terms[4])
        if terms[5] > ages[int(len(balances)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        tempList.append(terms[6])
        tempList.append(terms[7])
        if terms[8] == 'unknown':
            tempList.append('cellular')
        else:
            tempList.append(terms[8])
        if terms[9] > ages[int(len(days)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        tempList.append(terms[10])
        if terms[11] > ages[int(len(durations)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        if terms[12] > ages[int(len(campaigns)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        if terms[13] > ages[int(len(pdays)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        if terms[14] > ages[int(len(previous)/2)]:
            tempList.append('1')
        else:
            tempList.append('0')
        if terms[15] == 'unknown':
            tempList.append('failure')
        else:
            tempList.append(terms[15])
        tempList.append(terms[16])
        bank_train_data.append(tempList)


fa = '/home/nano/Downloads/山东中医药大学_王斌_1630_score.txt'
fa = '/home/nano/Documents/山东中医药大学_王文浩_1536/code/AI-Generated-image-identify/output/山东中医药大学_王文浩_1536_score.txt'
fb = '/home/nano/Downloads/山东中医药大学_吴镇东_1732_score.txt'
a = []
b = []
ad = {}
bd = {}
def get_num(f, arr):
    with open(f, 'r') as file:
        for i in file:
            if '秒' in i:
                break
            s = '_'
            e = '.'
            sa = i.index(s) + 1
            ea = i.index(e)
            t = int(i[sa: ea])
            arr.append(t)
    return arr


def get_dict(f):
    dic = {}
    with open(f, 'r') as file:
        count = 1
        for i in file:
            if '秒' in i or 'Total' in i:
                continue
            e = '\n'
            ea = i.index(e)
            if i[ea-1: ea] == 's':
                print(1)
            t = int(i[ea-1: ea])
            dic[count] = t
            count += 1
    return dic


# get_num(fa, a)
# get_num(fb, b)
# a.sort()
# b.sort()
ad = get_dict(fa)
bd = get_dict(fb)

def is_match_num(ai, bi):
    for i in range(1, 4001):
        if ai[i-1] != bi[i-1]:
            print(f'{i} not match')
        if i == 4000:
            print('all right')

def is_match_value(ai, bi):
    count = 0
    for i in range(1, 4001):
        if ai[i] != bi[i]:
            print(i, 'not match', bi[i])
            count += 1
        if i == 4000:
            print(f'count:{count}')

def count_1_0(ai):
    c0 = 0
    c1 = 0
    for i in range(1, 4001):
        if ai[i] == 1:
            c1 += 1
        if ai[i] == 0:
            c0 += 1
    print('0: ', c0, ' 1: ', c1)



is_match_value(ad, bd)

# print('/home/nano/Downloads/baidu/newest_test/test_2.jpg')

count_1_0(ad)
count_1_0(bd)

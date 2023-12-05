
# file_path_arr = os.listdir('/home/nano/PycharmProjects/AI-Generated-image-identify/output/山东中医药大学_王文浩_1443_score.txt')
obj = '/home/nano/Documents/山东中医药大学_王文浩_1536/code/AI-Generated-image-identify/output/山东中医药大学_王文浩_201_score.txt'
taget_file = '/home/nano/Documents/山东中医药大学_王文浩_1536/code/AI-Generated-image-identify/output/山东中医药大学_王文浩_201_score.txt'
a = []
with open(obj, 'r') as file:
    for i in file:
        a.append(i)
        t = 1

end = a[-1]
a = a[:-1]
b = {}
for i in a:
    s = '_'
    e = '.'
    sa = i.index(s)+1
    ea = i.index(e)
    t = int(i[sa: ea])
    b[t] = i

# sorted(b.keys())
with open(taget_file, 'w') as f:
    for i in range(1, 4001):
        f.write(b[i])
    f.write(end)

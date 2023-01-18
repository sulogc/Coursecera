import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#print(tf.__version__)
#print(dir(tf.keras))
#------------------------------------------------------------------------------------------
#multiple outputs 모델에 대해 araboza
# UCI의 Energy efficiency data set을 보자. 
# features: 8 , Labels:2 라서 multioutput model 쓰기 개꿀임.
# https://archive.ics.uci.edu/ml/datasets/Energy+efficiency

students = [
    '박해피',
    '이영희',
    '조민지',
    '조민지',
    '김철수',
    '이영희',
    '이영희',
    '김해킹',
    '박해피',
    '김철수',
    '한케이',
    '강디티',
    '조민지',
    '박해피',
    '김철수',
    '이영희',
    '박해피',
    '김해킹',
    '박해피',
    '한케이',
    '강디티',
]


dic = {}
for i in set(students):
    dic[i] = students.count(i)
l = sorted(dic.items(), key=lambda x: x[1], reverse=True)

for i in l:
    print(i[0])
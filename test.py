
# 아래의 명세를 읽고 Python 클래스를 활용하여 PublicTransport을 표현하시오.
# A.     PublicTransport는 이름 name 과 요금 fare을 인스턴스 속성으로 가진다.
# B.     탑승get_in, 하차get_off하는 메서드를 필요로 한다.    
#         i.          이 때, passenger의 수를 받는다.
# C.     현재 탑승자 수를 알 수 있어야 한다.
# D.     최종 수익을 계산하는 메소드 profit 은 요금과 전체 탑승자 수를 곱해서 계산한다. 

class PublicTransport:
#    passenger = 0
    def __init__(self, name, fare):
        self.name = name
        self.fare = fare
        self.passenger = 0
        self.all = 0 

    def get_in(self):
        self.passenger += 1
        self.all += 1

    def get_off(self):
        self.passenger -= 1
    
    def profit(self):
        return self.all * self.fare

# PT = PublicTransport('subway', 1250)
# print(PT.name, PT.fare)

# for _ in range(70):
#     PT.get_in()  
#     if _ % 3 == 1:
#         PT.get_off() 
#     print('_' * PT.passenger, PT.passenger)

# print(f'profit is ...{PT.profit()}')


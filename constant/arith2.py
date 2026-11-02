from decimal import Decimal as D, getcontext
getcontext().prec=40
for x in ["0.27658360966404055","0.27658360966","0.27658360688604","0.2765836278","0.27658360966592777"]:
    print(x, 1+1/D(x))
print(D("0.27658360966404055")-D("0.27658360966"))

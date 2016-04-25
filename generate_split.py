import json
import math
def generate_split(train_size,test_size):
	with open('price_list.json','r') as f:
		price_list = json.load(f)

	with open('train.txt','w') as f:
		for i in range(0,train_size):
			f.write('%d.jpg %d\n'% (i,math.ceil(price_list[i])))

	with open('test.txt', 'w') as f:
		for i in range(train_size,train_size+test_size):
			f.write('%d.jpg %d\n'%(i,math.ceil(price_list[i])))

generate_split(10,23)
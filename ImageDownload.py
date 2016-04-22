import gzip
import json
import urllib

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))
def load(path):
	with open(path,'r') as f:
		return json.load(f)

data = parse('./meta_Electronics.json.gz')

price_list = []
count = 0
for i in data:
	a = json.loads(i)
	try:
		price_list.append(a['price'])
		try:
			urllib.urlretrieve(a['imUrl'], "./images/%d.jpg"%count)
			count = count + 1
		except KeyError:
			print('keyError URL')
			price_list.pop(len(price_list)-1)
			continue
		except IOError:
			print('IOError')
			price_list.pop(len(price_list)-1)
			continue
		except:
			print("Unexpected Error")
			price_list.pop(len(price_list)-1)
			continue
	except KeyError:
		print('keyError Price')
		continue

with open('./price_list.json','w') as f:
	json.dump(price_list,f)
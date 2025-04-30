import pickle

with open('avgacc.pkl','rb') as f:
    a = pickle.load(f)

print(a)
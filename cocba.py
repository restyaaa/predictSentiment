import pickle

fasttext = 'models/cc.id.300.vec'

with open('models/cc.id.300.pkl', 'wb') as file:
    pickle.dump(fasttext, file)
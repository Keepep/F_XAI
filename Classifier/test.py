import pickle
class test():
    def __init__(self,model_name):

        self.trained_model_path = 'trained_model/' + model_name + '.sav'
    def test(self, te_data, te_label):
        model = pickle.load(open(self.trained_model_path, 'rb'))

        result = model.score(te_data, te_label)

        print 'Test Accuracy: {0:02f}'.format(result)
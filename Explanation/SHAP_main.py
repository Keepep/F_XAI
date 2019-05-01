from SHAP.explainers.kernel import KernelExplainer
from SHAP.plots.summary import  summary_plot
from SHAP.plots.force import *
import pickle
from Classifier.Data_processing import make_train_test
import matplotlib as plt
class shap_library():
    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model
    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        model = pickle.load(open(self.model, 'rb'))

        # the index of instances
        ins = 2
        print te_data.iloc[[ins]]
        explainer = KernelExplainer(model.predict_proba, tr_data, nsamples=100, link="logit")
        # Get shapley values
        shap_value = explainer.shap_values(te_data.iloc[[ins]], nsamples=100)

        plot=force_plot(explainer.expected_value[ins], shap_value[ins], te_data.iloc[[ins]], matplotlib=True, link="logit")

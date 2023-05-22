from django.shortcuts import render
from django.views.generic import TemplateView
# Create your views here.
from .source.logreg import MlModel
class MlModelView(TemplateView):

    template_name = "mlmodels/home.html"
    def get(self, request):
        news = "hello" #request.data.get("news")

        return render(request,self.template_name,{"resp": news})

    def post(self, request):
        news = request.POST.get("news")
        model = MlModel()
        resp = model.predict(news)
        return render(request,self.template_name,{"resp": resp})
from django.contrib import admin
from django.urls import path
from .views import MlModelView

app_name = "mlmodels"

urlpatterns = [
    path('', MlModelView.as_view(), name="mlmodel"),
]

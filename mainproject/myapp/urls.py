
# Create your views here.
from django.urls import path
from . import views
app_name = "myapp"

urlpatterns = [
	path("", views.index, name='index'),
	path("upload/", views.uploadCsv, name='upload'),
	path("upload/data/", views.plotData, name = 'plotData'),
	path("upload/trainComplete", views.trainCompleteData, name = 'trainComplete'),
	path("upload/trainPartial", views.trainPartialData, name = 'trainPartial'),
	path("upload/trainPartial/startTraining", views.startPartialTraining, name = 'startTraining'),
	path("/predict/", views.predict, name="predict"),
]
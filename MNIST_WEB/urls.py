from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('image/', views.handle_image),
    path('save/', views._save)
]

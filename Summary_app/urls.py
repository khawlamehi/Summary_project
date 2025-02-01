from django.urls import path
from . import views 

urlpatterns = [
    path('', views.index, name='index'),  # Cette URL renvoie à la page d'accueil (index.html)
    path('summarize/', views.summarize, name='summarize'),  # Cette URL appelle la vue pour générer le résumé
]

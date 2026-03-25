from django.urls import path
from . import views

urlpatterns = [
    # A raiz do site abre a lista de jogos (index.html)
    path('', views.pagina_inicial, name='pagina_inicial'),
    
    # A página de análise recebe parâmetros via GET
    path('analise/', views.analise_jogo, name='analise_jogo'),
]
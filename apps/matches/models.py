from django.db import models

class JogoDoDia(models.Model):
    data = models.DateField(db_index=True)
    hora = models.CharField(max_length=10, blank=True, null=True)
    liga = models.CharField(max_length=100, blank=True, null=True)
    home = models.CharField(max_length=100)
    away = models.CharField(max_length=100)
    odd_h = models.FloatField(blank=True, null=True)
    odd_d = models.FloatField(blank=True, null=True)
    odd_a = models.FloatField(blank=True, null=True)
    link_confronto = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name = "Jogo do Dia"
        verbose_name_plural = "Jogos do Dia"
        ordering = ['hora', 'liga']

    def __str__(self):
        return f"{self.data} - {self.home} vs {self.away}"

class Historico(models.Model):
    data = models.DateField(db_index=True)
    liga = models.CharField(max_length=100, blank=True, null=True)
    home = models.CharField(max_length=100)
    away = models.CharField(max_length=100)
    h_gols_ft = models.IntegerField(blank=True, null=True)
    a_gols_ft = models.IntegerField(blank=True, null=True)
    h_gols_ht = models.IntegerField(blank=True, null=True)
    a_gols_ht = models.IntegerField(blank=True, null=True)
    h_escanteios = models.IntegerField(blank=True, null=True)
    a_escanteios = models.IntegerField(blank=True, null=True)

    class Meta:
        verbose_name = "Histórico de Jogo"
        verbose_name_plural = "Histórico de Jogos"
        ordering = ['-data']

    def __str__(self):
        return f"{self.data} - {self.home} {self.h_gols_ft} x {self.a_gols_ft} {self.away}"


# Plano de Atualização e Deployment Seguro - redscore_django

Este documento descreve os procedimentos necessários para realizar a atualização do sistema redscore_django em ambiente de produção com máxima segurança.

## 1. Análise do Ambiente Atual
- **Framework:** Django 5.2.7
- **Banco de Dados:** SQLite (local) / PostgreSQL (via DATABASE_URL no Render/Supabase)
- **Servidor Web:** Gunicorn com WhiteNoise (arquivos estáticos)
- **Hospedagem:** Render (.onrender.com)
- **Dependências Críticas:** XGBoost, Scikit-learn, Pandas, Joblib (Alinhadas com o ambiente de treinamento redscore_ML)

## 2. Plano de Backup Completo
Antes de qualquer alteração, os seguintes backups devem ser realizados:
- **Backup de Arquivos:** Compactar a pasta raiz do projeto.
  ```bash
  tar -czvf backup_site_$(date +%F).tar.gz /home/rafael/Documentos/redscore_django
  ```
- **Backup de Banco de Dados:**
  - **SQLite:** Copiar `db.sqlite3`.
  - **PostgreSQL (Produção):** Realizar dump via CLI do provedor (ex: Supabase).
    ```bash
    pg_dump $DATABASE_URL > backup_db_$(date +%F).sql
    ```
- **Backup do Modelo:** Garantir cópia de `apps/matches/services/modelo_redscore_v2.pkl`.

## 3. Ambiente de Staging (Testes)
Configurar um ambiente idêntico à produção para validar as mudanças:
1. Criar um arquivo `.env.staging` (se necessário).
2. Instalar dependências em um novo ambiente virtual:
   ```bash
   pip install -r requirements.txt
   ```
3. Rodar as migrações:
   ```bash
   python manage.py migrate
   ```
4. Coletar arquivos estáticos:
   ```bash
   python manage.py collectstatic --no-input
   ```

## 4. Testes de Funcionalidades Críticas (Checklist)
Antes de ir para produção, os seguintes testes devem passar em Staging:
- [ ] Carregamento da página inicial sem erros.
- [ ] Carregamento da página de análise para confrontos específicos.
- [ ] Cálculo de probabilidades IA sem fallback (verificar log do terminal).
- [ ] Funcionamento do cache (estatísticas e features).
- [ ] Paridade de resultados com o ambiente de treinamento (usando `parity_check.py`).

## 5. Execução do Deployment (Produção)
Após aprovação em Staging:
1. Ativar modo manutenção no servidor (se disponível).
2. Atualizar o código (Git Pull ou Upload).
3. Executar o script de build:
   ```bash
   ./build.sh
   ```
4. Reiniciar o serviço do Gunicorn/Render.
5. Desativar modo manutenção.

## 6. Plano de Rollback
Caso ocorra um erro crítico (Erro 500, perda de dados):
1. Reverter para a versão anterior do código via Git ou Backup.
2. Restaurar o banco de dados a partir do dump:
   ```bash
   psql $DATABASE_URL < backup_db_YYYY-MM-DD.sql
   ```
3. Reiniciar o servidor.
4. Investigar a causa do erro nos logs (`LOGGING` configurado no `settings.py`).

## 7. Documentação e Logs
- Registrar a data e hora da atualização.
- Documentar qualquer erro encontrado durante o processo em Staging.
- Manter os arquivos de log de produção por pelo menos 7 dias após o deploy.

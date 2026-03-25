from pathlib import Path
import environ
import os

# -------------------------------------------------
# 🔧 Configuração base do Django
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Inicializa variáveis de ambiente (.env)
env = environ.Env()
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

# -------------------------------------------------
# 🔐 Segurança e debug
# -------------------------------------------------
SECRET_KEY = env("SECRET_KEY", default="dev-secret-key")
DEBUG = env.bool("DEBUG", default=True)
ALLOWED_HOSTS = env.list("ALLOWED_HOSTS", default=["localhost", "127.0.0.1", ".onrender.com"])

PASTA_JOGOS_DO_DIA = env("PASTA_JOGOS_DO_DIA", default="/home/rafael/Documentos/webscrapping_redscore/jogos_do_dia")
CAMINHO_HISTORICO = env("CAMINHO_HISTORICO", default="/home/rafael/Documentos/webscrapping_redscore/dados_redscore.csv")

# -------------------------------------------------
# 📦 Aplicações instaladas
# -------------------------------------------------
INSTALLED_APPS = [
    # Apps locais
    "apps.matches.apps.MatchesConfig",

    # Apps Django padrão
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
]


# -------------------------------------------------
# 🧱 Middlewares
# -------------------------------------------------
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "core.urls"

# -------------------------------------------------
# 🎨 Templates (HTML)
# -------------------------------------------------
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # opcional — templates globais
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "core.wsgi.application"

# -------------------------------------------------
# 💾 Banco de dados
# -------------------------------------------------
DATABASES = {
    "default": env.db("DATABASE_URL", default=f"sqlite:///{BASE_DIR}/db.sqlite3")
}

# -------------------------------------------------
# 🔒 Validação de senha
# -------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# -------------------------------------------------
# 🌎 Internacionalização
# -------------------------------------------------
LANGUAGE_CODE = "pt-br"
TIME_ZONE = "America/Sao_Paulo"
USE_I18N = True
USE_TZ = True

# -------------------------------------------------
# 🧭 Arquivos estáticos
# -------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'),]
STATIC_ROOT = BASE_DIR / "staticfiles"

# Whitenoise storage block caching
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# -------------------------------------------------
# 🔑 Configurações padrão
# -------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# -------------------------------------------------
# 📜 Configuração de logging
# -------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}


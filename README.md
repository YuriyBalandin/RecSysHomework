# Сервис рекомендаций (курс МТС)

## Запуск приложения

Зависимости в requirements.txt

### Python + Uvicorn

```
python main.py
```

Приложение запустится локально, в одном процессе. 
Хост и порт по умолчанию: `127.0.0.1` и `8080`.
Их можно изменить через переменные окружения `HOST` и `PORT`.

Управляет процессом легковесный [ASGI](https://asgi.readthedocs.io/en/latest/) server [uvicorn](https://www.uvicorn.org/).


Затем использовался cloudflare.

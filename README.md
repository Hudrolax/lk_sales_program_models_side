# lk_sales_program_models_side
Структура приложения:
https://github.com/Hudrolax/lk_sales_program_models_side.git - backend. Собирает данные из 1С, строит модели машинного обучения и прогнозы. Записывает данные в Redis.
https://github.com/Hudrolax/lk_sales_program.git - frontend. Собирает данные из 1С и Redis. Организует интерфейс через Dash.

### Install
1. Должен быть запущен Redis
2. клонировать репозиторий в /home/www/lk_sales_program_models_side
3. дать права на испольнение для всех *.sh файлов
4. Если необходимо, изменить entrypoint.sh (в нем параметры запуска приложения)
5. создать и заполнить env.py (структура в env_example.py)
6. выполнить ./build.sh (должен быть развернут Docker)
7. выполнить ./run_container.sh (запускает контейнер Docker с указанными в нем ключами)

### ключи Redis
```
actual_date
```
дата последней продажи в истории 1С
```
models_count
```
количество построенных моделей на данный момент
```
total_models_count
```
общее количество моделей к построению
```
ДД.ММ.ГГГГ,подразделение,регион,менеджер
```
структура JSON с таблицей для DASH. Структура загружается напрмую в dataframe Pandas.
  пример:
  ключ
```
31.07.2022,None,None,None
```
вернет структуру для основной таблицы в целом по компании
```
имя_модели,имя_группы,подразделение,регион,менеджер
```
вернет структуру JSON с прогнозом
  пример:
  ключ:
```
prophet,О-01.01. Фанера ФК,None,None,Березовский Юрий Юзикович
```
  структура возврата:
```
    {
        'name': model.name,
        'group': model.group,
        'subdivision': model.subdivision,
        'region': model.region,
        'manager': model.manager,
        'forecast': model.forecast.to_dict(), # pandas df в виде словаря
        'rmse': model.rmse,
        'figure': model.graph().to_json() # plotly figure в виде json
     }
```

ключ subdivision возвращает список подразделений в json
ключ region возвращает список регионов в json
ключ manager возвращает список менеджеров в json
Cписки содержат уникальные записи, сформированные по продажам за последний год.
формат json:
```
{
  "data": [массив и строк]
}
```

from datetime import datetime

import redis.exceptions

from env import SERVER_1C, BASE_1C, GET_QUERY_ROUTE, API_KEY, USER, PASSWORD
import requests
import json
from time import sleep
import logging
import pandas as pd
from queries import SALES_DATA_QUERY
from models import Models, Model
from myredis import MyRedis
from dateutil.relativedelta import relativedelta
from calendar import monthrange


# logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
def end_of_month(date: datetime) -> datetime:
    last_day_of_month = monthrange(date.year, date.month)[1]
    return date.replace(day=last_day_of_month, hour=23, minute=59, second=59, microsecond=9999)


class MyRedisConnectionError(Exception):
    def __init__(self, text):
        self.txt = text


class MyRedisModel(MyRedis):
    logger = logging.getLogger('MyRedisModel')

    def set(self, *args, **kwargs):
        try:
            super().set(*args, **kwargs)
        except redis.exceptions.ConnectionError as ex:
            raise MyRedisConnectionError(str(ex))

    def set_dict(self, key: str, _dict: dict, **kwargs) -> bool:
        """
        default expire is two months (5356800 sec)
        """
        self.logger.debug(f'redis_key: {key}')
        try:
            return super().set_dict(key, _dict, ex=5356800, **kwargs)
        except redis.exceptions.ConnectionError as ex:
            raise MyRedisConnectionError(str(ex))

    def save_forecast(self, model: Model) -> bool:
        """
        Сохраняет прогноз модели в Redis
        :param model: модель машинного обучения
        :return:
        """
        redis_key = f'{model.name},{model.group},{model.subdivision},{model.region},{model.manager}'
        model_dict = {
            'name': model.name,
            'group': model.group,
            'subdivision': model.subdivision,
            'region': model.region,
            'manager': model.manager,
            'forecast': model.forecast.to_dict(),
            'rmse': model.rmse,
            'figure': model.graph().to_json()
        }
        return self.set_dict(redis_key, model_dict)


class DataWorker:
    logger = logging.getLogger('DataWorker')

    def __init__(self, redis_host: str, redis_db: int, production: bool):
        # датафрейм для загрузки данных
        self._df = pd.DataFrame([], columns=['Группа', 'Период', 'Показатель', 'Подразделение, Регион, Менеджер'])

        # датафрейм для хранения очищенных данных
        self.dfc = pd.DataFrame([], columns=['Группа', 'Период', 'Показатель', 'Подразделение, Регион, Менеджер'])

        self.models = Models()

        self.session = requests.Session()
        self.session.auth = (USER, PASSWORD)
        self.redis = MyRedisModel(host=redis_host, db=redis_db)
        self.production = production

    @staticmethod
    def _date_to_str_1c(_date: datetime) -> str:
        return _date.strftime("%d.%m.%Y %H:%M:%S")

    @staticmethod
    def get_data(session, logger) -> dict:
        """
        Получает из 1С историю продаж в разрезе групп и подразделений помесячно
        :param session: сессия для подключения к 1C
        :param logger: логгер
        :return: dict с историей продаж для загрузки в dataframe
        """
        _empty_response = {"data": []}

        headers = {'Content-type': 'application/json',
                   'Accept': 'text/plain'}
        json_dict = {'api_key': API_KEY, 'query': SALES_DATA_QUERY}
        logger.debug(f'json_dict: {json_dict}')
        try:
            _route = f'http://{SERVER_1C}/{BASE_1C}{GET_QUERY_ROUTE}'
            logger.debug(f'route: {_route}')
            response_text = session.post(_route, json=json_dict, headers=headers).text
            # logger.debug(f'response_text: {response_text}')
            return json.loads(response_text)
        except requests.exceptions.ConnectTimeout as ex:
            logger.error(f'{requests.exceptions.ConnectTimeout}: {ex}')
        except requests.exceptions.ConnectionError as ex:
            logger.error(f'{requests.exceptions.ConnectionError}: {ex}')
        except json.decoder.JSONDecodeError as ex:
            logger.error(f'{json.decoder.JSONDecodeError}: {ex}')
        except Exception as ex:
            logger.critical(f'_get_data: {ex}')
            raise ex

        return _empty_response

    def load_data(self) -> None:
        """
        Загружает историю продаж из 1С в dataframe
        :return:
        """
        self.logger.info('Loading data from 1C...')
        _json_dict: dict = self.get_data(self.session, self.logger)
        try:
            self._df = pd.json_normalize(_json_dict['data'])
            # self._df.to_csv('history.csv', index=False)
            self.logger.debug('data updated')
        except KeyError as ex:
            self.logger.error(f'{KeyError}: {ex}')

    def preprocessing_data(self) -> None:
        """
        Выполняет препроцессинг загруженных данных.
        Обрезка лишних периодов, выбросов.
        :return:
        """
        self.logger.info('Preprocessing data...')
        if not self._df.empty:
            self._df['Период'] = pd.to_datetime(self._df['Период'], format='%d.%m.%Y %H:%M:%S')
            self.dfc = self._df[self._df['Период'] < datetime.now().replace(day=1, hour=0, minute=0, second=0, )]
            self.dfc = self.dfc.sort_values(by='Период', ascending=True, ignore_index=True)
            self.dfc['Регион'] = self.dfc['Регион'].fillna('Направление Краснодар+15км - Динской район')
            self.dfc['Подразделение'] = self.dfc['Подразделение'].fillna('Краснодар, Тополиная, 27/1')

            def empty_subdivision(x):
                if x == '':
                    return 'Краснодар, Тополиная, 27/1'
                else:
                    return x

            def empty_region(x):
                if x == '':
                    return 'Направление Краснодар+15км - Динской район'
                else:
                    return x

            self.dfc['Подразделение'] = self.dfc['Подразделение'].apply(empty_subdivision)
            self.dfc['Регион'] = self.dfc['Регион'].apply(empty_region)
            self.dfc.dropna()
            self.dfc = self.dfc[self.dfc['Группа'] != '']

    def save_models_count(self, models_count: int) -> None:
        self.redis.set('models_count', models_count)

    def _predict(self) -> None:
        self.logger.info('Make models and predictions...')
        if self._df.empty:
            return
        df_group = self.dfc.groupby(['Период', 'Группа'], as_index=False).sum()
        df_subdivision = self.dfc.groupby(['Период', 'Группа', 'Подразделение'], as_index=False).sum()
        df_region = self.dfc.groupby(['Период', 'Группа', 'Регион'], as_index=False).sum()
        df_manager = self.dfc.groupby(['Период', 'Группа', 'Менеджер'], as_index=False).sum()
        df_manager = df_manager.drop(df_manager[df_manager['Менеджер'] == ""].index)

        if not self.production:
            df_group = df_group[df_group['Группа'] == df_group['Группа'].unique()[0]]
            df_subdivision = df_subdivision[(df_subdivision['Группа'] == df_subdivision['Группа'].unique()[0]) \
                                            & (df_subdivision['Подразделение'] ==
                                               df_subdivision['Подразделение'].unique()[0])]
            df_region = df_region[(df_region['Группа'] == df_region['Группа'].unique()[0]) \
                                  & (df_region['Регион'] == df_region['Регион'].unique()[0])]
            df_manager = df_manager[(df_manager['Группа'] == df_manager['Группа'].unique()[0]) \
                                    & (df_manager['Менеджер'] == df_manager['Менеджер'].unique()[0])]

        last_year = datetime.now() - relativedelta(years=1)
        basic_models_count = len(df_group['Группа'].unique())
        subdivisions_models_count = len(df_subdivision['Группа'].unique())\
                                * len(df_subdivision[df_subdivision['Период'] > last_year]['Подразделение'].unique())
        regions_models_count = len(df_region['Группа'].unique())\
                               * len(df_region[df_region['Период'] > last_year]['Регион'].unique())
        managers_models_count = len(df_manager['Группа'].unique())\
                                * len(df_manager[df_manager['Период'] > last_year]['Менеджер'].unique())
        total_models_count = basic_models_count + subdivisions_models_count + regions_models_count + managers_models_count
        self.redis.set('total_models_count', total_models_count)
        print(f'Total models to build {total_models_count}')

        # модели в общем по-группам
        for i in self.models.make_fit_predict_raw_data(df_group):
            self.save_models_count(i)

        # модели в разрезе подразделений
        for i in self.models.make_fit_predict_raw_data(df_subdivision):
            self.save_models_count(i)

        # модели в разрезе регионов
        for i in self.models.make_fit_predict_raw_data(df_region):
            self.save_models_count(i)

        # модели в разрезе менеджеров
        for i in self.models.make_fit_predict_raw_data(df_manager):
            self.save_models_count(i)

    def df_list_to_save(self) -> list[pd.DataFrame]:
        """
        Функция возвращает список датафреймов с вариантами отображения в Dash
        :return: list of pd.DataFrame
        """
        df_list = []

        # df для групп по компании
        df = self.dfc.groupby(by=['Группа'], as_index=False).sum().copy()
        df = df.drop('Показатель', axis=1).sort_values(by='Группа')
        df_list.append(df)

        # df для групп по подразделениям
        df = self.dfc.groupby(by=['Группа', 'Подразделение'], as_index=False).sum().copy()
        df = df.drop('Показатель', axis=1).sort_values(by='Группа')
        df_list.append(df)

        # df для групп по регионам
        df = self.dfc.groupby(by=['Группа', 'Регион'], as_index=False).sum().copy()
        df = df.drop('Показатель', axis=1).sort_values(by='Группа')
        df_list.append(df)

        # df для групп по менеджерам
        df = self.dfc.groupby(by=['Группа', 'Менеджер'], as_index=False).sum().copy()
        df = df.drop('Показатель', axis=1).sort_values(by='Группа')
        df_list.append(df)

        # добавим колонки для прогноза
        for _df in df_list:
            _df['Прогноз'] = 0
            _df['RMSE'] = 0
        return df_list

    def save_forecasts(self) -> None:
        """
        Сохраняет прогнозы в Redis для всех моделей машинного обучения
        :return:
        """
        # сохраним предсказания моделей машинного обучения
        for model in self.models.models:
            self.redis.save_forecast(model)

    def save_dash_dataframes(self) -> None:
        """
        Сохраняет датафреймы для Dash в Redis
        :return:
        """
        self.logger.info('Save Dash dataframes to Redis...')

        def period_iterator():
            now_month = datetime.now()
            for i in range(6):
                yield end_of_month(now_month + relativedelta(months=i))

        def concat_forecast(_df, per: datetime) -> pd.DataFrame:
            kwargs = {
                'group': None,
                'subdivision': None,
                'region': None,
                'manager': None
            }
            for i in range(len(_df)):
                kwargs['group'] = _df.at[i, 'Группа']
                if 'Подразделение' in _df.columns:
                    kwargs['subdivision'] = _df.at[i, 'Подразделение']
                if 'Регион' in _df.columns:
                    kwargs['region'] = _df.at[i, 'Регион']
                if 'Менеджер' in _df.columns:
                    kwargs['manager'] = _df.at[i, 'Менеджер']
                _df.at[i, 'Прогноз'] = self.models.get_model(**kwargs).get_forecast(per)
                _df.at[i, 'RMSE'] = self.models.get_model(**kwargs).rmse
            return _df

        def key_format(period, subdivision=None, region=None, manager=None):
            return f'{period.strftime("%d.%m.%Y")},{subdivision},{region},{manager}'

        df_list = self.df_list_to_save()

        for _period in period_iterator():
            for _df2 in df_list:
                _df = concat_forecast(_df2.copy(), _period)

                if 'Подразделение' in _df.columns:
                    for _subdivision in _df['Подразделение'].unique():
                        df_save = _df[_df['Подразделение'] == _subdivision]
                        df_save = df_save.drop('Подразделение', axis=1)
                        key = key_format(_period, subdivision=_subdivision)
                        self.redis.set_dict(key, df_save.to_dict())

                elif 'Регион' in _df.columns:
                    for _region in _df['Регион'].unique():
                        df_save = _df[_df['Регион'] == _region]
                        df_save = df_save.drop('Регион', axis=1)
                        key = key_format(_period, region=_region)
                        self.redis.set_dict(key, df_save.to_dict())
                elif 'Менеджер' in _df.columns:
                    for _manager in _df['Менеджер'].unique():
                        df_save = _df[_df['Менеджер'] == _manager]
                        df_save = df_save.drop('Менеджер', axis=1)
                        key = key_format(_period, manager=_manager)
                        self.redis.set_dict(key, df_save.to_dict())
                else:
                    key = key_format(_period)
                    self.redis.set_dict(key, _df.to_dict())

    def save_options(self, option: str, key: str) -> None:
        options = self.dfc[self.dfc[option] != ""][option].unique().tolist()
        data = {
            'data': options
        }
        self.redis.set_dict(key, data)

    def save_actual_date(self) -> None:
        self.redis.set('actual_date', self.dfc['Период'].max().strftime("%d.%m.%Y"))

    def save_to_redis(self) -> None:
        self.save_forecasts()
        self.save_dash_dataframes()
        self.save_options('Подразделение', 'subdivision')
        self.save_options('Регион', 'region')
        self.save_options('Менеджер', 'manager')
        self.save_actual_date()

    def work_process(self):
        start_time = datetime.now()

        self.load_data()
        self.preprocessing_data()
        self._predict()
        self.save_to_redis()

        end_time = datetime.now()
        work_time = (end_time - start_time).total_seconds()
        self.redis.set('work_time', work_time)
        self.logger.info(f'work time {work_time} sec.')

    def run(self):
        while True:
            if self._df.empty or self._df['Период'].max() < datetime.now():
                try:
                    self.work_process()

                    self.logger.info('Work done!')
                except MyRedisConnectionError as ex:
                    self.logger.error(f'Redis connection error: {ex}')
                    print('sleep 10 second...')
                    sleep(10)

            if not self.production:
                break
            sleep_time = 86400
            self.logger.info(f'sleep for {sleep_time}')
            for i in range(sleep_time):
                last_date = None
                try:
                    last_date = self.redis.get('actual_date')
                except redis.exceptions.ConnectionError as ex:
                    sleep(1)
                    continue

                if last_date is not None:
                    sleep(1)
                else:
                    self.logger.warning("Perhaps Redis server is rebooted. I'l try update all data.")
                    break

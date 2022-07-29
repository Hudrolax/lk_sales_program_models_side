import logging
import prophet
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
import pdb
import numpy as np
from prophet.plot import plot_plotly
import plotly.express as px


prophet.forecaster.logger.setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


class Model:
    logger = logging.getLogger(__name__)

    def __init__(self, group: str, name: str | None = None, subdivision: str | None = None, region: str | None = None,
                 manager: str | None = None, periods: int = 6, freq: str = 'M'):
        """
        :param name: Имя модели машинного обучения (если None - пустая модель)
        :param group: группа НоменклатурыЛК
        :param subdivision: подразделение, если нет, то None
        :param region: регион, если нет, то None
        :param manager: ЗаказКлиента.Менеджер, если нет, то None
        :param periods: periods for prediction
        :param freq: prediction frequency
                    The frequency can be anything from the pandas list of frequency strings here:
                     https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        """
        self.name = name
        self.group = group
        self.subdivision = subdivision
        self.region = region
        self.manager = manager
        self.periods = periods
        self.freq = freq

        # model object
        self.model = None
        # model forecast
        self.forecast = None
        # mse
        self.mse = None
        # model rmse
        self.rmse = None
        # saved model filename
        self.model_filename = None

        if self.name is None:
            def df_generator():
                now = datetime.now()
                for i in range(self.periods):
                    date = now + relativedelta(months=i)
                    day_of_week, days = calendar.monthrange(date.year, date.month)
                    yield [date.replace(day=days, hour=23, minute=59, second=59, microsecond=999999), 0]

            self.forecast = pd.DataFrame([k for k in df_generator()], columns=['ds', 'yhat'])
            self.rmse = 0
            self.mse = 0

    def graph(self):
        if type(self.model) == prophet.Prophet:
            return plot_plotly(self.model, self.forecast, trend=True)
        else:
            return px.scatter()

    def get_forecast(self, period: datetime) -> float:
        series = self.forecast[(self.forecast['ds'].dt.month == period.month) & \
                               (self.forecast['ds'].dt.year == period.year)]
        forecast = 0
        if len(series) == 1:
            forecast = round(float(series['yhat']), 3)
        return forecast

    def make(self, **kwargs) -> None:
        """
        make model
        :param
            if model is "prophet": no kwargs
        :return: None
        """
        if self.name == 'prophet':
            self.logger.info(f'Make model for group:{kwargs.get("group")}, subdivision:{kwargs.get("subdivision")},'
                             f'region:{kwargs.get("region")}, manager:{kwargs.get("manager")}')
            self.model = prophet.Prophet()
            logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
        else:
            raise Exception('Model name error')

    def fit(self, **kwargs):
        """
        Fit model
        :param kwargs:
            if model is "prophet":
                "df": dataframe for prophet model
        :return:
        """
        if self.name == 'prophet':
            if 'df' in kwargs:
                self.model.fit(kwargs['df'])
            else:
                raise Exception('"df" dataframe expected')
        else:
            raise Exception('Model name error')

    def predict(self, **kwargs):
        """
        make predict
        """
        if self.name == 'prophet':
            future = self.model.make_future_dataframe(periods=self.periods, freq=self.freq)
            self.forecast = self.model.predict(future)
            history = self.model.history.groupby(by='ds').sum()['y']
            se = np.square(self.forecast[:len(history)].groupby(by='ds').sum()['yhat'] - history)
            self.mse = np.mean(se)
            self.rmse = np.sqrt(self.mse)
        else:
            raise Exception('Model name error')

    def make_fit_predict(self, **kwargs) -> None:
        self.make(**kwargs)
        self.fit(**kwargs)
        self.predict(**kwargs)

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.group == other.group and self.subdivision == other.subdivision and self.region == other.region \
                   and self.manager == other.manager
        else:
            return False


class Models:
    logger = logging.getLogger(__name__)

    def __init__(self):
        """
        self.models: list of class Model
        """
        self.models = []

    def get_model(self, group: str, **kwargs) -> Model:
        for _model in self.models:
            if _model == Model(group, name='same', **kwargs):
                return _model
        return Model(group, **kwargs)

    def add_model(self, model) -> None:
        _model: Model
        for _model in self.models:
            if _model == model:
                _model = model
                return
        self.models.append(model)

    def make_fit_predict_raw_data(self, df: pd.DataFrame) -> None:
        """
        Решает, какую модель обучения выбрать для исходных данных, подготавливает данные, создает модель, обучает,
         делает прогноз
        :param df: сгруппированный датафрейм с историческими данными
        :return:
        """
        last_year = datetime.now() - relativedelta(years=1)
        # прогноз для подразделений
        if 'Подразделение' in df.columns:
            actual_subd = df[df['Период'] > last_year]['Подразделение'].unique()
            for group in df['Группа'].unique():
                for subdivision in actual_subd:
                    df_model = df[(df['Группа'] == group) & (df['Подразделение'] == subdivision)].sort_values(
                        by='Период')
                    df_model = df_model.drop(['Группа', 'Подразделение', 'Ед'], axis=1).rename(columns={'Период': 'ds',
                                                                                                        'Показатель': 'y'})
                    # pdb.set_trace()
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', subdivision=subdivision)
                        model.make_fit_predict(group=group, subdivision=subdivision, df=df_model)
                    else:
                        model = Model(group, subdivision=subdivision)
                    self.add_model(model)

        # прогноз для регионов
        elif 'Регион' in df.columns:
            actual_regions = df[df['Период'] > last_year]['Регион'].unique()
            for group in df['Группа'].unique():
                for region in actual_regions:
                    df_model = df[(df['Группа'] == group) & (df['Регион'] == region)].sort_values(
                        by='Период')
                    df_model = df_model.drop(['Группа', 'Регион', 'Ед'], axis=1).rename(
                        columns={'Период': 'ds',
                                 'Показатель': 'y'})
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', region=region)
                        model.make_fit_predict(group=group, region=region, df=df_model)
                    else:
                        model = Model(group, region=region)
                    self.add_model(model)

        # прогноз для менеджеров
        elif 'Менеджер' in df.columns:
            actual_managers = df[df['Период'] > last_year]['Менеджер'].unique()
            for group in df['Группа'].unique():
                for manager in actual_managers:
                    df_model = df[(df['Группа'] == group) & (df['Менеджер'] == manager)].sort_values(
                        by='Период')
                    df_model = df_model.drop(['Группа', 'Менеджер', 'Ед'], axis=1).rename(
                        columns={'Период': 'ds',
                                 'Показатель': 'y'})
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', manager=manager)
                        model.make_fit_predict(group=group, manager=manager, df=df_model)
                    else:
                        model = Model(group, manager=manager)
                    self.add_model(model)

        # прогноз для компании в целом
        else:
            for group in df['Группа'].unique():
                df_model = df[df['Группа'] == group].sort_values(by='Период')
                df_model = df_model.drop(['Группа', 'Ед'], axis=1).rename(columns={'Период': 'ds', 'Показатель': 'y'})
                if len(df_model) > 1:
                    model = Model(group, name='prophet')
                    model.make_fit_predict(group=group, subdivision='По компании в целом', df=df_model)
                else:
                    model = Model(group)
                self.add_model(model)

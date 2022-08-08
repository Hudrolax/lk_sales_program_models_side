import logging
import prophet
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
import pdb
import numpy as np
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objs as go

prophet.forecaster.logger.setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


class Model:
    logger = logging.getLogger(__name__)

    def __init__(self, group: str, name: str | None = None, subdivision: str | None = None, region: str | None = None,
                 manager: str | None = None, periods: int = 6, freq: str = 'M', logistic: bool = True,
                 cap_percent: float = 0.2, quantile_dev_multi: float = 1.5, drop_outliers: bool = True):
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
        :param logistic: bool Использовать логистические уровни максимума и минимума
        :param cap_percent: int процент logistic cap от максимального значения. Default 20 %.
        :param quantile_dev_multi: float множитель отклонения от 25 и 75 персентилей для определения выбросов
        :param drop_outliers: bool Удалять ли выбросы из истории
        """
        self.name = name
        self.group = group
        self.subdivision = subdivision
        self.region = region
        self.manager = manager
        self.periods = periods
        self.freq = freq
        self.logistic = logistic
        self.cap_percent = cap_percent
        self.quantile_dev_multi = quantile_dev_multi
        self.drop_outliers = drop_outliers

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

        # Нижний лимит выборосов
        self.lower_limit = None
        # Верхний лимит выбросов
        self.upper_limit = None
        # df с выбросами (колонки ds, y)
        self.outliers = pd.DataFrame()
        # df с историей для прогноза (колонки df, y и если logistic, то cap, floor)
        self.df = pd.DataFrame()
        # df неочищенный от выбросов с историей для прогноза (колонки df, y)
        self.raw_df = pd.DataFrame()

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
            fig = plot_plotly(self.model, self.forecast, trend=True)
            fig.update_layout(
                title="Прогноз",
                xaxis_title="Период",
                yaxis_title="Объем продаж",
            )
            fig.update_layout(xaxis_rangeslider_visible=False)
            if not self.outliers.empty:
                fig.add_trace(go.Scatter(x=self.outliers['ds'], y=self.outliers['y'],
                                         mode='markers',
                                         name='outliers',
                                         marker_color='red'))
            return fig
        else:
            return px.scatter()

    def graph_component(self):
        if type(self.model) == prophet.Prophet:
            fig = plot_components_plotly(self.model, self.forecast)
            fig.update_layout(
                title="Компоненты прогноза",
                width=800,
                height=300
            )
            return fig
        else:
            return px.scatter()

    def boxplot(self):
        if type(self.model) == prophet.Prophet and not self.raw_df.empty:
            fig = px.box(self.raw_df['y'], width=400, height=300, points="all")
            fig.update_layout(
                title="Распределение объемов продаж",
                xaxis_title="Распределение",
                yaxis_title="Объем продаж",
            )
            return fig
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
            if self.logistic:
                growth = 'logistic'
            else:
                growth = 'linear'
            self.model = prophet.Prophet(growth=growth)
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
                self.raw_df = kwargs['df'].copy()
                # drop outliers
                if self.drop_outliers:
                    self.lower_limit = self.raw_df.y.quantile(0.25) - (self.raw_df.y.quantile(0.75) - self.raw_df.y.quantile(0.25)) * self.quantile_dev_multi
                    self.upper_limit = self.raw_df.y.quantile(0.75) + (self.raw_df.y.quantile(0.75) - self.raw_df.y.quantile(0.25)) * self.quantile_dev_multi
                    self.outliers = self.raw_df[(self.raw_df['y'] < self.lower_limit) | (self.raw_df['y'] > self.upper_limit)]
                    self.df = self.raw_df[(self.raw_df['y'] >= self.lower_limit) & (self.raw_df['y'] <= self.upper_limit)].copy()

                if self.logistic:
                    self.df['cap'] = self.df.y.max() + self.df.y.max() * self.cap_percent
                    self.df['floor'] = 0
                self.model.fit(self.df)
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
            if self.logistic:
                future['cap'] = self.df.y.max() + self.df.y.max() * self.cap_percent
                future['floor'] = 0
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

    def __str__(self):
        return f'''model {self.name}, group {self.group}, subdivision {self.subdivision}, region {self.region},
               manager {self.manager}'''


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

    def make_fit_predict_raw_data(self, df: pd.DataFrame):
        """
        Решает, какую модель обучения выбрать для исходных данных, подготавливает данные, создает модель, обучает,
         делает прогноз.
        Функцию необходимо вызывать как итератор, т.к. она возвращает количество созданных моделей на каждой итерации
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
                    df_model = df_model[['Период', 'Показатель']].rename(columns={'Период': 'ds', 'Показатель': 'y'})
                    # pdb.set_trace()
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', subdivision=subdivision)
                        model.make_fit_predict(group=group, subdivision=subdivision, df=df_model)
                    else:
                        model = Model(group, subdivision=subdivision)
                    self.add_model(model)
                    yield len(self.models)

        # прогноз для регионов
        elif 'Регион' in df.columns:
            actual_regions = df[df['Период'] > last_year]['Регион'].unique()
            for group in df['Группа'].unique():
                for region in actual_regions:
                    df_model = df[(df['Группа'] == group) & (df['Регион'] == region)].sort_values(
                        by='Период')
                    df_model = df_model[['Период', 'Показатель']].rename(columns={'Период': 'ds', 'Показатель': 'y'})
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', region=region)
                        model.make_fit_predict(group=group, region=region, df=df_model)
                    else:
                        model = Model(group, region=region)
                    self.add_model(model)
                    yield len(self.models)

        # прогноз для менеджеров
        elif 'Менеджер' in df.columns:
            actual_managers = df[df['Период'] > last_year]['Менеджер'].unique()
            for group in df['Группа'].unique():
                for manager in actual_managers:
                    df_model = df[(df['Группа'] == group) & (df['Менеджер'] == manager)].sort_values(
                        by='Период')
                    df_model = df_model[['Период', 'Показатель']].rename(columns={'Период': 'ds', 'Показатель': 'y'})
                    if len(df_model) > 1:
                        model = Model(group, name='prophet', manager=manager)
                        model.make_fit_predict(group=group, manager=manager, df=df_model)
                    else:
                        model = Model(group, manager=manager)
                    self.add_model(model)
                    yield len(self.models)

        # прогноз для компании в целом
        else:
            for group in df['Группа'].unique():
                df_model = df[df['Группа'] == group].sort_values(by='Период')
                df_model = df_model[['Период', 'Показатель']].rename(columns={'Период': 'ds', 'Показатель': 'y'})
                if len(df_model) > 1:
                    model = Model(group, name='prophet')
                    model.make_fit_predict(group=group, df=df_model)
                else:
                    model = Model(group)
                self.add_model(model)
                yield len(self.models)

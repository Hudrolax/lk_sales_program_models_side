import data_worker
import logging
import pandas as pd
import pytest
import pdb
from dateutil.relativedelta import relativedelta
from queries import SALES_DATA_QUERY

logger = logging.getLogger(__name__)


@pytest.fixture(scope='session')
def test_setup():
    dw = data_worker.DataWorker('192.168.19.18', 15, False)
    return dw


def test_get_data(test_setup):
    """
    Тест получения сырых данных из 1С
    :param test_setup:
    :return:
    """
    data = test_setup.get_data(test_setup.session, test_setup.logger, SALES_DATA_QUERY)
    assert data['data'] != []


def test_load_data_from_1c(test_setup):
    """
    Тест загрузки в датафрейм данных из 1С
    :param test_setup:
    :return:
    """
    test_setup.load_data()
    assert type(test_setup._df) == pd.DataFrame
    assert 'Период' in test_setup._df.columns
    assert 'Показатель' in test_setup._df.columns
    assert 'Подразделение' in test_setup._df.columns
    assert 'Группа' in test_setup._df.columns
    assert 'Регион' in test_setup._df.columns
    assert 'Менеджер' in test_setup._df.columns

    assert not test_setup._df.empty
    test_setup.preprocessing_data()


def test_get_empty_model(test_setup):
    """
    Тест возврата пустой модели, если не найдена модель для запрошенных данных
    :param test_setup:
    :return:
    """
    model = test_setup.models.get_model(group='Неизвестная группа', subdivision='Непонятное подразделение')
    assert model.name is None
    assert type(model.forecast) == pd.DataFrame
    period = test_setup.dfc['Период'].max() + relativedelta(months=1)
    # pdb.set_trace()
    forecast = model.get_forecast(period)
    assert forecast == 0
    assert model.mse == model.rmse == model.get_forecast(period)


def test_make_fit_predict_raw_data(test_setup):
    """
    Тест создания модели на основе выборки из сырых данных
    :param test_setup:
    :return:
    """
    df_group = test_setup.dfc.groupby(['Период', 'Группа', 'Ед'], as_index=False).sum()
    df_group = df_group[df_group['Группа'] == df_group['Группа'].unique()[0]]

    df_subdivision = test_setup.dfc.groupby(['Период', 'Группа', 'Подразделение', 'Ед'], as_index=False).sum()
    df_subdivision = df_subdivision[(df_subdivision['Группа'] == df_subdivision['Группа'].unique()[0]) \
                                    & (df_subdivision['Подразделение'] == df_subdivision['Подразделение'].unique()[0])]

    df_region = test_setup.dfc.groupby(['Период', 'Группа', 'Регион', 'Ед'], as_index=False).sum()
    df_region = df_region[(df_region['Группа'] == df_region['Группа'].unique()[0]) \
                                    & (df_region['Регион'] == df_region['Регион'].unique()[0])]

    df_manager = test_setup.dfc.groupby(['Период', 'Группа', 'Менеджер', 'Ед'], as_index=False).sum()
    df_manager = df_manager.drop(df_manager[df_manager['Менеджер'] == ""].index)
    df_manager = df_manager[(df_manager['Группа'] == df_manager['Группа'].unique()[0]) \
                          & (df_manager['Менеджер'] == df_manager['Менеджер'].unique()[0])]

    # модели в общем по-группам
    for i in test_setup.models.make_fit_predict_raw_data(df_group):
        pass

    # модели в разрезе подразделений
    for i in test_setup.models.make_fit_predict_raw_data(df_subdivision):
        pass

    # модели в разрезе регионов
    for i in test_setup.models.make_fit_predict_raw_data(df_region):
        pass

    # модели в разрезе менеджеров
    for i in test_setup.models.make_fit_predict_raw_data(df_manager):
        pass

    assert len(test_setup.models.models) > 0
    period = test_setup.dfc['Период'].max() + relativedelta(months=1)
    for i in range(4):
        assert test_setup.models.models[i].name is not None
        assert test_setup.models.models[i].rmse > 0
        # pdb.set_trace()
        assert type(test_setup.models.models[i].forecast) == pd.DataFrame
        forecast = test_setup.models.models[i].get_forecast(period)
        assert forecast != 0
        assert test_setup.models.models[i].mse != test_setup.models.models[0].rmse


def test_save_to_redis(test_setup):
    test_setup.save_to_redis()

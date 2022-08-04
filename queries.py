SALES_DATA_QUERY="""
ВЫБРАТЬ
	Товары.Ссылка КАК Номенклатура,
	ВЫБОР
		КОГДА Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.01. Фанера ФК"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.02. Фанера ФСФ береза"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.03. Фанера ФСФ хвойная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.04. Фанера ФЛФ (ламинированная)"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.05. OSB"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.06. ДВП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.07. ДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.08. ЛДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-01.09. МДФ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.01. Вагонка"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.02. Имитация бруса, Блок - Хаус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.03. Доска пола"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.04. Террасная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.05. Планкен"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.06. Круглый погонаж ель/сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.07.01. Брус строганый"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.07.02. Рейка/ Брусок"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.07.04. Доска Строганая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.07.06. Доска обрезная сухая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.08. Импрегнированные и термомодифицированные изделия"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.09. Наличник, Плинтус, Уголок, Притвор, Штапик"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-02.10. Наличник, Плинтус, Уголок, Притвор, Штапик лиственница"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-03.01. Дверная коробка сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-03.02. Клееный брус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-03.03. Мебельный щит и подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-03.04. Дверка жалюзийная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-03.05. Подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-07. СИСТЕМЫ ХРАНЕНИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-08.ФАСАДЫ и ДВЕРКИ ДЛЯ МЕБЕЛИ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-04. ЛЕСТНИЧНЫЕ ЭЛЕМЕНТЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-05. БАНИ, САУНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-06. НЕКОНДИЦИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "О-07.  ПРОДУКЦИЯ из ЭКЗОТИЧЕСКИХ ПОРОД ДРЕВЕСИНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-06. ДВЕРИ МЕЖКОМНАТНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-01. МАТЕРИАЛЫ ЛАКОКРАСОЧНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-02. КРЕПЕЖ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-03. ИЗДЕЛИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-04. ЗАЩИТНЫЕ ПОКРЫТИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Наименование = "С-05. ИНСТРУМЕНТ И ИНВЕНТАРЬ"
			ТОГДА Товары.ЛК_ГруппаНоменклатуры.Родитель
		КОГДА Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.01. Фанера ФК"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.02. Фанера ФСФ береза"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.03. Фанера ФСФ хвойная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.04. Фанера ФЛФ (ламинированная)"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.05. OSB"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.06. ДВП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.07. ДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.08. ЛДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.09. МДФ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.01. Вагонка"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.02. Имитация бруса, Блок - Хаус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.03. Доска пола"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.04. Террасная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.05. Планкен"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.06. Круглый погонаж ель/сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.07.01. Брус строганый"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.07.02. Рейка/ Брусок"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.07.04. Доска Строганая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.07.06. Доска обрезная сухая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.08. Импрегнированные и термомодифицированные изделия"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.09. Наличник, Плинтус, Уголок, Притвор, Штапик"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-02.10. Наличник, Плинтус, Уголок, Притвор, Штапик лиственница"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-03.01. Дверная коробка сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-03.02. Клееный брус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-03.03. Мебельный щит и подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-03.04. Дверка жалюзийная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-03.05. Подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-07. СИСТЕМЫ ХРАНЕНИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-08.ФАСАДЫ и ДВЕРКИ ДЛЯ МЕБЕЛИ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-04. ЛЕСТНИЧНЫЕ ЭЛЕМЕНТЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-05. БАНИ, САУНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-06. НЕКОНДИЦИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-07.  ПРОДУКЦИЯ из ЭКЗОТИЧЕСКИХ ПОРОД ДРЕВЕСИНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-06. ДВЕРИ МЕЖКОМНАТНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-01. МАТЕРИАЛЫ ЛАКОКРАСОЧНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-02. КРЕПЕЖ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-03. ИЗДЕЛИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-04. ЗАЩИТНЫЕ ПОКРЫТИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "С-05. ИНСТРУМЕНТ И ИНВЕНТАРЬ"
			ТОГДА Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель
		КОГДА Товары.ЛК_ГруппаНоменклатуры.Родитель.Родитель.Наименование = "О-01.01. Фанера ФК"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.02. Фанера ФСФ береза"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.03. Фанера ФСФ хвойная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.04. Фанера ФЛФ (ламинированная)"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.05. OSB"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.06. ДВП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.07. ДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.08. ЛДСП"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-01.09. МДФ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.01. Вагонка"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.02. Имитация бруса, Блок - Хаус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.03. Доска пола"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.04. Террасная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.05. Планкен"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.06. Круглый погонаж ель/сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.07.01. Брус строганый"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.07.02. Рейка/ Брусок"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.07.04. Доска Строганая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.07.06. Доска обрезная сухая"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.08. Импрегнированные и термомодифицированные изделия"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.09. Наличник, Плинтус, Уголок, Притвор, Штапик"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-02.10. Наличник, Плинтус, Уголок, Притвор, Штапик лиственница"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-03.01. Дверная коробка сосна"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-03.02. Клееный брус"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-03.03. Мебельный щит и подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-03.04. Дверка жалюзийная"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-03.05. Подоконная доска"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-07. СИСТЕМЫ ХРАНЕНИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-08.ФАСАДЫ и ДВЕРКИ ДЛЯ МЕБЕЛИ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-04. ЛЕСТНИЧНЫЕ ЭЛЕМЕНТЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-05. БАНИ, САУНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-06. НЕКОНДИЦИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "О-07.  ПРОДУКЦИЯ из ЭКЗОТИЧЕСКИХ ПОРОД ДРЕВЕСИНЫ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-06. ДВЕРИ МЕЖКОМНАТНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-01. МАТЕРИАЛЫ ЛАКОКРАСОЧНЫЕ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-02. КРЕПЕЖ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-03. ИЗДЕЛИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-04. ЗАЩИТНЫЕ ПОКРЫТИЯ"
				ИЛИ Товары.ЛК_ГруппаНоменклатуры.Наименование = "С-05. ИНСТРУМЕНТ И ИНВЕНТАРЬ"
			ТОГДА Товары.ЛК_ГруппаНоменклатуры
		ИНАЧЕ ""
	КОНЕЦ КАК Группа
ПОМЕСТИТЬ ГруппыНоменклатурыВТ
ИЗ
	Справочник.Номенклатура КАК Товары
ГДЕ
	Товары.ЛК_ГруппаНоменклатуры.Наименование <> ""
;

////////////////////////////////////////////////////////////////////////////////
ВЫБРАТЬ
	ВЫБОР
		КОГДА БизнесРегионы.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Ссылка
		КОГДА БизнесРегионы.Родитель.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Родитель
		КОГДА БизнесРегионы.Родитель.Родитель.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Родитель.Родитель
		КОГДА БизнесРегионы.Родитель.Родитель.Родитель.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Родитель.Родитель.Родитель
		КОГДА БизнесРегионы.Родитель.Родитель.Родитель.Родитель.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Родитель.Родитель.Родитель.Родитель
		КОГДА БизнесРегионы.Родитель.Родитель.Родитель.Родитель.Родитель.Родитель = ЗНАЧЕНИЕ(Справочник.БизнесРегионы.ПустаяСсылка)
			ТОГДА БизнесРегионы.Родитель.Родитель.Родитель.Родитель.Родитель
		ИНАЧЕ ""
	КОНЕЦ КАК РегионРодитель,
	БизнесРегионы.Ссылка КАК Регион
ПОМЕСТИТЬ БизнесРегионыВТ
ИЗ
	Справочник.БизнесРегионы КАК БизнесРегионы
;

////////////////////////////////////////////////////////////////////////////////
ВЫБРАТЬ
	ЕСТЬNULL(ГруппыНоменклатурыВТ.Группа, "") КАК Группа,
	КОНЕЦПЕРИОДА(ПартииЛескрафт.Период, МЕСЯЦ) КАК Период,
	СУММА(ВЫБОР
			КОГДА ПартииЛескрафт.Номенклатура.ЕдиницаДляОтчетов.Наименование = "м3"
				ТОГДА ВЫБОР
						КОГДА ПартииЛескрафт.Номенклатура.ОбъемЗнаменатель = 0
							ТОГДА 0
						ИНАЧЕ ПартииЛескрафт.Количество * ПартииЛескрафт.Номенклатура.ОбъемЧислитель / ПартииЛескрафт.Номенклатура.ОбъемЗнаменатель
					КОНЕЦ
			ИНАЧЕ ПартииЛескрафт.Оборот
		КОНЕЦ) КАК Показатель,
	ПартииЛескрафт.ЗаказКлиента.Менеджер КАК Менеджер,
	ПартииЛескрафт.Склад.Подразделение КАК Подразделение,
	БизнесРегионыВТ.РегионРодитель КАК Регион,
	МАКСИМУМ(ВЫБОР
			КОГДА ПартииЛескрафт.Номенклатура.ЕдиницаДляОтчетов.Наименование = "м3"
				ТОГДА "м3"
			ИНАЧЕ "руб"
		КОНЕЦ) КАК Ед
ИЗ
	РегистрНакопления.ПартииЛескрафт КАК ПартииЛескрафт
		ЛЕВОЕ СОЕДИНЕНИЕ ГруппыНоменклатурыВТ КАК ГруппыНоменклатурыВТ
		ПО ПартииЛескрафт.Номенклатура = ГруппыНоменклатурыВТ.Номенклатура
		ЛЕВОЕ СОЕДИНЕНИЕ БизнесРегионыВТ КАК БизнесРегионыВТ
		ПО ПартииЛескрафт.ПокупательПартнер.БизнесРегион = БизнесРегионыВТ.Регион
ГДЕ
	ПартииЛескрафт.Оборот <> 0
	И ПартииЛескрафт.Период >= ДАТАВРЕМЯ(2016, 8, 1, 0, 0, 0)

СГРУППИРОВАТЬ ПО
	ЕСТЬNULL(ГруппыНоменклатурыВТ.Группа, ""),
	КОНЕЦПЕРИОДА(ПартииЛескрафт.Период, МЕСЯЦ),
	ПартииЛескрафт.ЗаказКлиента.Менеджер,
	ПартииЛескрафт.Склад.Подразделение,
	БизнесРегионыВТ.РегионРодитель
"""

WORKING_MANAGERS = """
ВЫБРАТЬ
	Мотивация_РаботникиСрезПоследних.Пользователь КАК Пользователь
ИЗ
	РегистрСведений.Мотивация_Работники.СрезПоследних КАК Мотивация_РаботникиСрезПоследних
ГДЕ
	Мотивация_РаботникиСрезПоследних.Должность.Наименование = "Менеджер по продажам"
	И Мотивация_РаботникиСрезПоследних.ПричинаИзмененияСостояния <> ЗНАЧЕНИЕ(Перечисление.Мотивация_ПричиныИзмененияСостояния.Увольнение)

УПОРЯДОЧИТЬ ПО
	Пользователь
АВТОУПОРЯДОЧИВАНИЕ
"""
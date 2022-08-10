#!/bin/bash

# -p --production=True // false - прогноз будет только для 4х первых значений. Для продакшина указывать true
# -rh --redis_host="172.17.0.1" // IP Redis. 172.17.0.1 - стандарный IP внешнего хоста Docker
# -rdb --redis_db=0 // номер базы данных Redis. 0 - дефолт.

python main.py -p=true -rh="172.17.0.1" -rdb=1
#!/bin/bash

docker run \
-d \
--name lk_sales_program_models_side \
--restart unless-stopped \
-v /app:/home/www/lk_sales_program_models_side \
lk_sales_program_models_side
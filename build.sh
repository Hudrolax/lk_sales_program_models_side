#!/bin/bash

docker stop lk_sales_program_models_side
docker rm lk_sales_program_models_side
docker rmi lk_sales_program_models_side
docker build . -t lk_sales_program_models_side
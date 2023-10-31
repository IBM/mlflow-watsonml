import logging
import sys

LOGGER = logging.getLogger("mlflow-watsonml")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.FileHandler("app.log"))

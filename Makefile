split:
	python data/splitDataset.py

train:
	python src/train.py

predict:
	python src/predict.py

sysTest:
	python app.py
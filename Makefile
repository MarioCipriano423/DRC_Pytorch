start:
	python app.py

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

train:
	python train_model.py

test:
	pytest tests/

clean:
	rm -rf __pycache__ *.pyc *.log

greet:
	echo "👋 ¡Hola desde tu entorno PyTorch en Docker!"

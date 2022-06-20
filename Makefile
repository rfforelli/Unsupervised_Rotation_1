all:
	conda env create -f environment.yml
	
clean:
	rm -rf my-hls-test
	rm -rf my-hls-test.tar.gz
	rm -rf keras_figures
	rm -rf hls_figures
	rm -rf __pycache__

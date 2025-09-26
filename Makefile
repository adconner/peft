experiments=$(patsubst configs/%.yaml,data/%.out,$(wildcard configs/*.yaml))

all: $(experiments)

data/%.out: configs/%.yaml
	python train.py --config_path $<
	
